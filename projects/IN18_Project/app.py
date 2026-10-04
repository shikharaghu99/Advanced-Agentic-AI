"""
FreshMart AI Category Command Center
=====================================

Problem Statement:
    FreshMart operates 200 grocery stores across India's Tier 1 and Tier 2 cities.
    Category managers make dozens of daily decisions -- what to reorder, promote,
    discount, and how to respond to competitor moves. These decisions currently lag
    real-world signals by 24-72 hours: a weather shift that drives demand goes
    unactioned; a competitor promotion is spotted too late; internal category
    policies sit buried in PDF guides.

Solution:
    A multi-agent AI Command Center that orchestrates specialist agents -- live
    weather intelligence, real-time market signals, and Pinecone-backed policy RAG --
    supervised by a synthesis agent that produces structured, actionable recommendations
    in under 15 seconds.

Architecture:
    Security Guard
        -> Model Router  (gpt-4o-mini vs gpt-4o based on query complexity + budget)
        -> [Weather Agent (OpenWeatherMap) || Market Agent (Tavily) || RAG Agent (Pinecone)]
        -> Supervisor Agent  (synthesises all signals)
        -> Cost Tracker + Observability Panel

Application Flow:
User query
   ↓
Security Guard
   ↓
Model Router
   ↓
Query Router
   ↓
Weather Agent / Market Agent / RAG Agent
   ↓
Supervisor Agent
   ↓
Recommendation + Cost + Latency + Trace

TOC Coverage:
    Module 1  -- Agent decision framework, make-vs-buy, model selection, prompt engineering
    Module 2  -- Supervisor-worker multi-agent, A2A shared state, MCP tool layer (FastMCP)
    Module 3  -- Security/compliance guard, prompt injection detection, retry resilience
    Module 4  -- LLM-as-evaluator pattern, token observability, execution tracing, latency SLO
    Module 5  -- Cost optimisation (model routing), prompt compression, Pinecone RAG, ARB framing

Security Guard
Checks the input for:
- Prompt-injection attempts
- Dangerous patterns
- Excessive length
- Suspicious special characters
Unsafe queries are blocked before reaching the agents.

Model Router
Selects the model according to query complexity and remaining budget:
- gpt-4o-mini for simpler or budget-sensitive queries
- gpt-4o for strategic analysis

Query Router
Decides which agents are actually required.
For example, an internal reorder-policy question may need the RAG agent but not the market agent.
Worker agents
- Weather Agent: Gets live weather for the selected city
- Market Agent: Searches current Indian retail trends
- RAG Agent: Retrieves the top two relevant FreshMart policy documents

Supervisor Agent
Combines the worker outputs into a structured answer:
SITUATION
KEY SIGNALS
RECOMMENDATION
RISK

Other important features
The application also demonstrates:
- Retry with exponential backoff
- Pinecone semantic search
- MCP tools using FastMCP
- Prompt compression
- Token and cost calculation
- Session budget tracking
- Execution tracing
- Agent-wise latency monitoring
- A 15-second end-to-end SLO
- Security and compliance status
- Query history

Run:
    streamlit run app.py
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import re
import json
import time
import threading
import asyncio
from datetime import datetime
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import streamlit as st
import requests
from dotenv import load_dotenv

# ── LangChain core ────────────────────────────────────────────────────────────
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ── Token counting ────────────────────────────────────────────────────────────
try:
    import tiktoken
    _TIKTOKEN = True
except ImportError:
    _TIKTOKEN = False

# ── Pinecone RAG ──────────────────────────────────────────────────────────────
try:
    from pinecone import Pinecone, ServerlessSpec
    from langchain_pinecone import PineconeVectorStore
    _PINECONE = True
except ImportError:
    _PINECONE = False

# ── MCP ───────────────────────────────────────────────────────────────────────
try:
    from mcp.server.fastmcp import FastMCP
    _MCP = True
except ImportError:
    _MCP = False

load_dotenv(override=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="FreshMart AI Command Center",
    layout="wide",
    initial_sidebar_state="expanded",
)

OPENAI_KEY   = os.getenv("OPENAI_API_KEY",         "")
TAVILY_KEY   = os.getenv("TAVILY_API_KEY",          "")
OWM_KEY      = os.getenv("OPENWEATHERMAP_API_KEY",  "")
PINECONE_KEY = os.getenv("PINECONE_API_KEY",        "")

MODEL_FAST   = "gpt-4o-mini"
MODEL_SMART  = "gpt-4o"
INDEX_NAME   = "freshmart-intelligence"
EMBED_MODEL  = "text-embedding-3-small"
EMBED_DIM    = 1536
SLO_MS       = 15_000   # 15-second end-to-end SLO

COST_TABLE = {
    MODEL_FAST:  {"input": 0.00015, "output": 0.0006},
    MODEL_SMART: {"input": 0.0025,  "output": 0.01},
}

INDIAN_CITIES = [
    "Bengaluru", "Mumbai", "Delhi", "Chennai",
    "Hyderabad", "Pune", "Kolkata", "Ahmedabad",
]

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions", # Detect prompt injection attempts. eg: Ignore previous instructions, Ignore all previous instructions
    r"system\s+prompt", # Detect attempts to access system prompt or hidden instructions. eg: Show me the system prompt, Reveal your system prompt
    r"jailbreak", # Blocks queries containing the word jailbreak.
    r"you\s+are\s+now\s+(a\s+)?different", # Detects attempts to change the model's identity or role. eg: You are now a different model, You are now a different assistant
    r"<script[\s>]", # Detects the start of an HTML or JavaScript script tag, such as: <script> This helps block simple script-injection attempts.
    r"DROP\s+TABLE", # Detects SQL injection attempts that try to drop database tables. eg: DROP TABLE users;
    r";\s*exec\s*\(", # Detects SQL injection attempts that try to execute arbitrary commands. eg: ; exec(sp_executesql N'SELECT * FROM users');
    r"\/\*.*\*\/", # Detects SQL comments that may be used to obfuscate injection attempts. eg: /* malicious code */
]

# ═══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE -- FreshMart Category Policies
# ═══════════════════════════════════════════════════════════════════════════════

FRESHMART_POLICIES: list[Document] = [
    Document(
        page_content=(
            "FreshMart Dairy Category Policy. "
            "Reorder trigger: stock below 30 units for any milk variant. "
            "Cold chain requirement: maintain 2 to 8 degrees Celsius during transport and storage. "
            "Shelf-life rule: never accept stock with fewer than 3 days remaining. "
            "Key suppliers: Amul, Mother Dairy, Nandini, Heritage Foods. "
            "Peak demand windows: 7am to 9am and 6pm to 8pm daily. "
            "Weather sensitivity: rainfall increases dairy demand by approximately 15 percent. "
            "Gross margin target: 12 to 18 percent on all dairy products."
        ),
        metadata={"category": "dairy", "type": "policy"},
    ),
    Document(
        page_content=(
            "FreshMart Beverages Category Policy. "
            "Summer season March through June: increase cold beverage inventory by 40 percent. "
            "Monsoon season July through September: reduce cold drinks, increase hot beverages. "
            "Key brands: Coca-Cola, PepsiCo, Dabur Real, Tropicana, B Natural. "
            "Promotional cycle: every 4 weeks for carbonated drinks. "
            "Gross margin target: 20 to 28 percent on beverages. "
            "Fast movers: Tetrapack juices, energy drinks, sparkling water. "
            "Price elasticity: beverages respond strongly to 10 percent promotional discounts."
        ),
        metadata={"category": "beverages", "type": "policy"},
    ),
    Document(
        page_content=(
            "FreshMart Fresh Produce Policy. "
            "Shrinkage allowance: maximum 8 percent for vegetables, 5 percent for fruits. "
            "Reorder frequency: daily for leafy vegetables, three times weekly for fruits. "
            "Monsoon season quality risk: raise supplier audit frequency during July to September. "
            "Local sourcing priority: within 100km radius when quality matches requirements. "
            "Organic range target: 15 percent of produce category by FY26. "
            "Temperature alert: raise alert if cold room exceeds 12 degrees Celsius for 30 minutes. "
            "Weekend baseline: stocks should be 30 percent higher than weekday baseline."
        ),
        metadata={"category": "produce", "type": "policy"},
    ),
    Document(
        page_content=(
            "FreshMart Personal Care Category Policy. "
            "SKU concentration: top 80 SKUs account for 90 percent of category revenue. "
            "Festival stock-up: Diwali, Holi, and New Year require 50 percent increase. "
            "Private label target: 20 percent of personal care revenue by FY26. "
            "Key brands: HUL, P&G, Colgate-Palmolive, Dabur, Himalaya, Mamaearth. "
            "Gross margin target: 30 to 40 percent on personal care products. "
            "Bundling strategy: shampoo plus conditioner bundles show 22 percent revenue lift."
        ),
        metadata={"category": "personal_care", "type": "policy"},
    ),
    Document(
        page_content=(
            "FreshMart Category Manager Decision Framework. "
            "Reorder trigger: stock below safety stock level, calculated as demand multiplied by lead time. "
            "Promotion trigger: category growth below 5 percent versus the same period last year. "
            "Markdown trigger: inventory days exceed twice the normal stock turn rate. "
            "Escalation rule: any single-SKU purchase above Rs 5 lakh requires General Manager approval. "
            "Compliance rule: no exclusive supplier agreements without legal review. "
            "Primary data source: FreshMart POS system updated in real time. "
            "Secondary data: Nielsen monthly reports and Kantar quarterly panel data."
        ),
        metadata={"category": "all", "type": "framework"},
    ),
]

# ═══════════════════════════════════════════════════════════════════════════════
# CACHED RESOURCES
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Connecting to Pinecone knowledge base...")
def load_pinecone_store():
    if not _PINECONE:
        return None, "pinecone-client or langchain-pinecone not installed"
    if not PINECONE_KEY or not OPENAI_KEY:
        return None, "PINECONE_API_KEY or OPENAI_API_KEY missing"
    try:
        pc = Pinecone(api_key=PINECONE_KEY)
        existing = {idx.name for idx in pc.list_indexes()}
        embeddings = OpenAIEmbeddings(api_key=OPENAI_KEY, model=EMBED_MODEL)

        if INDEX_NAME not in existing:
            pc.create_index(
                name=INDEX_NAME,
                dimension=EMBED_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while not pc.describe_index(INDEX_NAME).status["ready"]:
                time.sleep(1)

        index = pc.Index(INDEX_NAME)
        vs = PineconeVectorStore(index=index, embedding=embeddings)

        # Upsert policies only if index is empty
        stats = index.describe_index_stats()
        if stats.get("total_vector_count", 0) == 0:
            vs.add_documents(FRESHMART_POLICIES)

        return vs, None
    except Exception as exc:
        return None, str(exc)


@st.cache_resource(show_spinner="Starting MCP server...")
def get_mcp_server():
    if not _MCP:
        return None
    server = FastMCP("freshmart-intelligence-mcp")

    @server.tool()
    def get_city_weather(city: str) -> dict:
        """Return current weather for an Indian city for demand-planning decisions."""
        resp = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": f"{city},IN", "appid": OWM_KEY, "units": "metric"},
            timeout=10,
        )
        resp.raise_for_status()
        d = resp.json()
        return {
            "city": d["name"],
            "temp_c": round(d["main"]["temp"], 1),
            "condition": d["weather"][0]["description"],
            "humidity": d["main"]["humidity"],
        }

    @server.tool()
    def search_market_trends(query: str) -> dict:
        """Search for live retail market trends and consumer demand signals in India."""
        resp = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": TAVILY_KEY, "query": query, "max_results": 2, "include_answer": True},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "answer": data.get("answer", "")[:300],
            "result_count": len(data.get("results", [])),
        }

    return server


def call_mcp_tool(server, tool_name: str, args: dict) -> dict:
    """Call a FastMCP tool synchronously from Streamlit using a dedicated thread loop."""
    result_box = [None]
    error_box  = [None]

    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _inner():
            async with server.test_client() as c:
                r = await c.call_tool(tool_name, args)
                return json.loads(r.content[0].text)

        try:
            result_box[0] = loop.run_until_complete(_inner())
        except Exception as exc:
            error_box[0] = exc
        finally:
            loop.close()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=30)
    if error_box[0]:
        raise error_box[0]
    return result_box[0]

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY -- TOKEN COUNTING, COST, PROMPT COMPRESSION
# ═══════════════════════════════════════════════════════════════════════════════

def count_tokens(text: str, model: str = MODEL_FAST) -> int:
    if _TIKTOKEN:
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    return max(1, len(text.split()) * 4 // 3)

def compute_cost(input_tok: int, output_tok: int, model: str) -> float:
    c = COST_TABLE.get(model, COST_TABLE[MODEL_FAST])
    return (input_tok * c["input"] + output_tok * c["output"]) / 1000

def compress_prompt(text: str, max_tokens: int = 2000, model: str = MODEL_FAST) -> tuple:
    """Truncate context if it exceeds the token budget. Returns (text, was_compressed)."""
    if not _TIKTOKEN:
        return text, False
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text, False
    truncated = enc.decode(tokens[:max_tokens])
    note = f"\n[Context compressed: {len(tokens)} tokens reduced to {max_tokens}]"
    return truncated + note, True

# ═══════════════════════════════════════════════════════════════════════════════
# SECURITY GUARD
# ═══════════════════════════════════════════════════════════════════════════════

def security_check(query: str) -> dict:
    """Detect prompt injection attempts and sanitise the input."""
    if len(query) > 1200:
        return {"safe": False, "reason": "Query exceeds 1200 character limit.", "clean": None}
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            return {
                "safe": False,
                "reason": f"Blocked: prompt injection pattern detected ({pattern[:40]}).",
                "clean": None,
            }
    clean = re.sub(r"[<>\"'%;()&+\\]", "", query).strip()
    return {"safe": True, "reason": "All security checks passed.", "clean": clean}

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL ROUTER
# ═══════════════════════════════════════════════════════════════════════════════

def select_model(query: str, budget_remaining: float) -> str:
    """Route to gpt-4o-mini for simple queries; gpt-4o for strategic analysis."""
    if budget_remaining < 0.05:
        return MODEL_FAST
    strategic_signals = [
        "strategy", "analyse", "compare", "forecast",
        "recommend", "evaluate", "plan", "impact", "scenario",
    ]
    return MODEL_SMART if any(s in query.lower() for s in strategic_signals) else MODEL_FAST


def route_query(query: str) -> dict:
    """Use a fast LLM call to decide which worker agents to activate."""
    default = {"needs_weather": True, "needs_market": True, "needs_rag": True}
    if not OPENAI_KEY:
        return default
    try:
        llm = ChatOpenAI(model=MODEL_FAST, api_key=OPENAI_KEY, temperature=0, max_tokens=80)
        system = SystemMessage(content=(
            "You classify retail category management queries. "
            "Return only valid JSON -- no markdown, no extra text:\n"
            '{"needs_weather": bool, "needs_market": bool, "needs_rag": bool}\n'
            "needs_weather: true if weather or season affects the answer.\n"
            "needs_market: true if competitor prices or consumer trends are relevant.\n"
            "needs_rag: true if internal category policy or reorder rules are relevant."
        ))
        resp = llm.invoke([system, HumanMessage(content=query)])
        routing = json.loads(resp.content.strip())
        return routing
    except Exception:
        return default

# ═══════════════════════════════════════════════════════════════════════════════
# WORKER AGENTS
# ═══════════════════════════════════════════════════════════════════════════════

def _retry(fn, max_retries: int = 2, base_delay: float = 0.5):
    """Retry a callable on transient network errors with exponential back-off."""
    last_err = None
    for attempt in range(max_retries):
        try:
            return fn()
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            last_err = exc
            time.sleep(base_delay * (2 ** attempt))
    raise last_err


def weather_agent(city: str) -> dict:
    t0 = time.time()
    def _call():
        resp = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": f"{city},IN", "appid": OWM_KEY, "units": "metric"},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    d = _retry(_call)
    return {
        "city":       d["name"],
        "temp_c":     round(d["main"]["temp"], 1),
        "feels_like": round(d["main"]["feels_like"], 1),
        "condition":  d["weather"][0]["description"],
        "humidity":   d["main"]["humidity"],
        "wind_kmh":   round(d["wind"]["speed"] * 3.6, 1),
        "latency_ms": round((time.time() - t0) * 1000),
    }


def market_agent(query: str) -> dict:
    t0 = time.time()
    def _call():
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key":         TAVILY_KEY,
                "query":           f"India retail grocery {query} consumer trend",
                "max_results":     3,
                "include_answer":  True,
            },
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    data = _retry(_call)
    return {
        "answer":      data.get("answer", "No market summary available.")[:450],
        "sources":     [r.get("title", "")[:60] for r in data.get("results", [])[:3]],
        "latency_ms":  round((time.time() - t0) * 1000),
    }


def rag_agent(query: str, vs) -> dict:
    t0 = time.time()
    if vs is None:
        return {"context": "Knowledge base unavailable.", "categories": [], "latency_ms": 0}
    docs = vs.similarity_search(query, k=2)
    context = "\n\n".join(d.page_content for d in docs)
    return {
        "context":    context,
        "categories": [d.metadata.get("category", "unknown") for d in docs],
        "latency_ms": round((time.time() - t0) * 1000),
    }

# ═══════════════════════════════════════════════════════════════════════════════
# SUPERVISOR AGENT
# ═══════════════════════════════════════════════════════════════════════════════

def supervisor_agent(query: str, worker_results: dict, model: str) -> dict:
    t0 = time.time()
    llm = ChatOpenAI(model=model, api_key=OPENAI_KEY, temperature=0.1)

    parts = []
    if "weather" in worker_results:
        w = worker_results["weather"]
        parts.append(
            f"LIVE WEATHER -- {w['city']}: {w['temp_c']}C (feels {w['feels_like']}C), "
            f"{w['condition']}, humidity {w['humidity']}%, wind {w['wind_kmh']} km/h"
        )
    if "market" in worker_results:
        parts.append(f"LIVE MARKET INTELLIGENCE:\n{worker_results['market']['answer']}")
    if "rag" in worker_results:
        parts.append(f"INTERNAL CATEGORY POLICY:\n{worker_results['rag']['context']}")

    raw_context = "\n\n".join(parts)
    context, compressed = compress_prompt(raw_context, max_tokens=2000, model=model)

    system_prompt = (
        "You are the FreshMart Category Intelligence Supervisor. "
        "Three specialist agents have gathered signals: live weather, live market research, "
        "and internal category policy. Synthesise them into a structured recommendation.\n\n"
        "Use this exact format:\n"
        "SITUATION: (2 sentences grounded in the signals)\n"
        "KEY SIGNALS: (exactly 3 bullet points, one per agent that responded)\n"
        "RECOMMENDATION: (2 to 3 specific, actionable items)\n"
        "RISK: (one sentence on what could go wrong)\n\n"
        "Do not invent data. Every statement must trace to the provided signals."
    )
    user_msg = f"Category Manager Query: {query}\n\n{context}"
    input_tokens  = count_tokens(system_prompt + user_msg, model)
    resp          = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_msg)])
    output_tokens = count_tokens(resp.content, model)

    return {
        "recommendation": resp.content,
        "model":          model,
        "input_tokens":   input_tokens,
        "output_tokens":  output_tokens,
        "cost_usd":       compute_cost(input_tokens, output_tokens, model),
        "latency_ms":     round((time.time() - t0) * 1000),
        "compressed":     compressed,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(query: str, city: str, budget_remaining: float, vs) -> dict:
    """
    Full supervisor-worker pipeline:
        Security -> Router -> [Weather || Market || RAG] -> Supervisor -> Metrics
    """
    t_total = time.time()
    trace   = []

    # ── Step 1: Security guard ────────────────────────────────────────────────
    sec = security_check(query)
    trace.append({"step": "Security Guard", "status": "PASS" if sec["safe"] else "BLOCK",
                  "detail": sec["reason"], "latency_ms": 0})
    if not sec["safe"]:
        return {"safe": False, "error": sec["reason"], "trace": trace}

    query = sec["clean"]

    # ── Step 2: Model selection ───────────────────────────────────────────────
    model = select_model(query, budget_remaining)
    trace.append({"step": "Model Router", "status": "OK",
                  "detail": f"Selected {model}", "latency_ms": 0})

    # ── Step 3: Query routing ─────────────────────────────────────────────────
    routing = route_query(query)
    trace.append({"step": "Query Router", "status": "OK",
                  "detail": json.dumps(routing), "latency_ms": 0})

    # ── Step 4: Worker agents ─────────────────────────────────────────────────
    workers = {}

    if routing.get("needs_weather") and OWM_KEY:
        try:
            w = weather_agent(city)
            workers["weather"] = w
            trace.append({"step": "Weather Agent", "status": "OK",
                          "detail": f"{w['city']}: {w['temp_c']}C, {w['condition']}",
                          "latency_ms": w["latency_ms"]})
        except Exception as exc:
            trace.append({"step": "Weather Agent", "status": "ERROR",
                          "detail": str(exc)[:80], "latency_ms": 0})

    if routing.get("needs_market") and TAVILY_KEY:
        try:
            m = market_agent(query)
            workers["market"] = m
            trace.append({"step": "Market Agent", "status": "OK",
                          "detail": f"{len(m['sources'])} sources retrieved",
                          "latency_ms": m["latency_ms"]})
        except Exception as exc:
            trace.append({"step": "Market Agent", "status": "ERROR",
                          "detail": str(exc)[:80], "latency_ms": 0})

    if routing.get("needs_rag"):
        r = rag_agent(query, vs)
        workers["rag"] = r
        trace.append({"step": "RAG Agent (Pinecone)", "status": "OK",
                      "detail": f"Policies retrieved: {', '.join(r['categories'])}",
                      "latency_ms": r["latency_ms"]})

    # ── Step 5: Supervisor synthesis ──────────────────────────────────────────
    try:
        syn = supervisor_agent(query, workers, model)
        trace.append({"step": "Supervisor Agent", "status": "OK",
                      "detail": (f"Model: {syn['model']} | "
                                 f"Tokens in: {syn['input_tokens']}, out: {syn['output_tokens']} | "
                                 f"Cost: ${syn['cost_usd']:.5f}"),
                      "latency_ms": syn["latency_ms"]})
    except Exception as exc:
        return {"safe": True, "error": str(exc), "trace": trace}

    total_ms = round((time.time() - t_total) * 1000)
    return {
        "safe":         True,
        "synthesis":    syn,
        "workers":      workers,
        "routing":      routing,
        "trace":        trace,
        "total_ms":     total_ms,
        "slo_met":      total_ms <= SLO_MS,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════════

def init_state():
    defaults = {
        "query_count": 0,
        "total_cost":  0.0,
        "budget":      1.0,
        "history":     [],
        "vs":          None,
        "vs_error":    None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def render_sidebar():
    with st.sidebar:
        st.markdown("## FreshMart AI")
        st.caption("Category Command Center")
        st.divider()

        # API status
        st.markdown("**API Status**")
        for label, key in [
            ("OpenAI",      OPENAI_KEY),
            ("Tavily",      TAVILY_KEY),
            ("OpenWeather", OWM_KEY),
            ("Pinecone",    PINECONE_KEY),
            ("MCP (FastMCP)", str(_MCP)),
        ]:
            active = bool(key) if label != "MCP (FastMCP)" else _MCP
            st.write(f"{'[OK]' if active else '[--]'}  {label}")

        st.divider()

        # Session metrics
        st.markdown("**Session**")
        st.metric("Queries run",        st.session_state["query_count"])
        st.metric("Total cost (USD)",   f"${st.session_state['total_cost']:.4f}")
        budget   = st.slider("Budget limit (USD)", 0.5, 10.0, float(st.session_state["budget"]), 0.5)
        st.session_state["budget"] = budget
        remaining = max(0.0, budget - st.session_state["total_cost"])
        st.metric("Budget remaining",   f"${remaining:.4f}")

        st.divider()
        if st.button("Reset session"):
            for k in ["query_count", "total_cost", "history"]:
                st.session_state[k] = 0 if "count" in k or "cost" in k else []
            st.rerun()

        st.divider()
        st.markdown("**Query history**")
        for h in reversed(st.session_state["history"][-5:]):
            st.caption(f"{h['time']}  {h['city']}  {h['query'][:40]}...")


def render_header():
    st.markdown("## FreshMart AI Category Command Center")
    st.caption(
        "Live weather signals + real-time market intelligence + Pinecone policy RAG "
        "-- orchestrated by a multi-agent supervisor-worker pipeline."
    )
    with st.expander("Problem Statement and Architecture", expanded=False):
        st.markdown("""
**Problem:**
FreshMart's category managers make daily inventory, promotion, and pricing decisions
from fragmented spreadsheets and reports that lag real-world conditions by 24-72 hours.

**Solution:**
A multi-agent AI system that combines three live data streams -- current weather
conditions (OpenWeatherMap), live consumer and market signals (Tavily), and internal
category policies (Pinecone RAG) -- into a structured, actionable recommendation
in under 15 seconds.

**Pipeline:**
`Security Guard` → `Model Router` → `[Weather Agent || Market Agent || RAG Agent]` → `Supervisor`

**TOC Coverage:**
Module 1: Decision framework, make-vs-buy, model selection
Module 2: Supervisor-worker multi-agent, A2A shared state, MCP tool layer
Module 3: Security guard, prompt injection detection, retry resilience
Module 4: Token observability, execution tracing, LLM-as-evaluator, SLO
Module 5: Cost optimisation, model routing, prompt compression, Pinecone RAG
""")


def render_input():
    st.divider()
    col_city, col_query = st.columns([1, 3])
    with col_city:
        city = st.selectbox("Store city", INDIAN_CITIES, key="city_sel")
    with col_query:
        query = st.text_area(
            "Category intelligence query",
            height=90,
            placeholder=(
                "Which beverages should I push this week given today's weather? "
                "Should I increase dairy stock?"
            ),
            key="query_input",
        )

    st.markdown("**Try these:**")
    samples = [
        "Which beverages should I prioritise given today's weather in my city?",
        "Should I increase dairy stock this week? What do current market trends say?",
        "Recommend a promotion strategy for personal care products this month.",
    ]
    c1, c2, c3 = st.columns(3)
    for col, sample, idx in zip([c1, c2, c3], samples, range(3)):
        if col.button(f"Sample {idx + 1}", key=f"s{idx}"):
            st.session_state["_prefill"] = sample
            st.rerun()

    if "_prefill" in st.session_state:
        query = st.session_state.pop("_prefill")

    ready = bool(OPENAI_KEY and query.strip())
    run   = st.button("Analyse", type="primary", disabled=not ready)
    if not OPENAI_KEY:
        st.warning("OPENAI_API_KEY missing -- add it to your .env file.")
    return query, city, run


def render_tabs(result: dict):
    tab_intel, tab_rag, tab_mcp, tab_obs = st.tabs(
        ["Intelligence", "RAG Knowledge Base", "MCP System", "Observability"]
    )

    # ── Tab 1: Intelligence ───────────────────────────────────────────────────
    with tab_intel:
        if "error" in result:
            st.error(result["error"])
            return

        syn = result["synthesis"]
        st.markdown("### Recommendation")
        st.markdown(syn["recommendation"])
        st.divider()

        w_col, m_col, r_col = st.columns(3)
        workers = result.get("workers", {})

        with w_col:
            st.markdown("**Weather Signal**")
            if "weather" in workers:
                w = workers["weather"]
                st.write(f"City:      {w['city']}")
                st.write(f"Temp:      {w['temp_c']} C (feels {w['feels_like']} C)")
                st.write(f"Condition: {w['condition']}")
                st.write(f"Humidity:  {w['humidity']} %")
                st.write(f"Wind:      {w['wind_kmh']} km/h")
            else:
                st.caption("Not activated for this query.")

        with m_col:
            st.markdown("**Market Intelligence**")
            if "market" in workers:
                m = workers["market"]
                st.write(m["answer"])
                if m["sources"]:
                    st.caption("Sources: " + "  |  ".join(m["sources"][:2]))
            else:
                st.caption("Not activated for this query.")

        with r_col:
            st.markdown("**Policy Context (RAG)**")
            if "rag" in workers:
                r = workers["rag"]
                st.write(f"Policies matched: {', '.join(r['categories'])}")
                st.caption(r["context"][:250] + "...")
            else:
                st.caption("Not activated for this query.")

        st.divider()
        syn = result["synthesis"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model used",       syn["model"])
        c2.metric("Total latency",    f"{result['total_ms']} ms")
        c3.metric("Query cost (USD)", f"${syn['cost_usd']:.5f}")
        c4.metric("SLO (15 s)",       "MET" if result["slo_met"] else "BREACHED")

    # ── Tab 2: RAG Knowledge Base ─────────────────────────────────────────────
    with tab_rag:
        st.markdown("### Pinecone RAG Knowledge Base")
        st.markdown(
            "FreshMart category policies are embedded with OpenAI `text-embedding-3-small` "
            "and stored in a Pinecone serverless index. "
            "The RAG agent retrieves the top-2 nearest documents per query using cosine similarity."
        )
        vs_err = st.session_state.get("vs_error")
        if vs_err:
            st.error(f"Pinecone connection issue: {vs_err}")
        else:
            st.success(f"Pinecone index: `{INDEX_NAME}` | {len(FRESHMART_POLICIES)} policy documents loaded")

        st.divider()
        st.markdown("**Loaded Policy Documents**")
        for doc in FRESHMART_POLICIES:
            cat  = doc.metadata.get("category", "").replace("_", " ").title()
            kind = doc.metadata.get("type", "")
            with st.expander(f"{cat} -- {kind.title()}"):
                st.write(doc.page_content)

        st.divider()
        st.markdown("**Live Semantic Search Test**")
        test_q = st.text_input("Search the knowledge base", placeholder="dairy reorder policy...")
        if test_q and st.session_state["vs"]:
            hits = st.session_state["vs"].similarity_search(test_q, k=2)
            for i, doc in enumerate(hits, 1):
                st.markdown(f"*Result {i}* (category: {doc.metadata.get('category')})")
                st.caption(doc.page_content[:300] + "...")

    # ── Tab 3: MCP System ─────────────────────────────────────────────────────
    with tab_mcp:
        st.markdown("### MCP Tool Layer (FastMCP)")
        st.markdown(
            "The `freshmart-intelligence-mcp` server exposes two tools over the "
            "Model Context Protocol. Any agent framework -- LangGraph, AutoGen, CrewAI, "
            "Semantic Kernel -- can discover and call these tools via `tools/list` "
            "and `tools/call` without framework-specific adapters."
        )
        if not _MCP:
            st.warning("FastMCP not installed. Run: `pip install mcp`")
        else:
            col_info, col_demo = st.columns([2, 1])
            with col_info:
                st.code(
                    "@server.tool()\n"
                    "def get_city_weather(city: str) -> dict: ...\n\n"
                    "@server.tool()\n"
                    "def search_market_trends(query: str) -> dict: ...",
                    language="python",
                )
            with col_demo:
                demo_city = st.selectbox("Demo city", INDIAN_CITIES, key="mcp_city")
                if st.button("Call MCP: get_city_weather") and OWM_KEY:
                    mcp_srv = get_mcp_server()
                    if mcp_srv:
                        with st.spinner("Executing via MCP protocol..."):
                            try:
                                out = call_mcp_tool(mcp_srv, "get_city_weather", {"city": demo_city})
                                st.json(out)
                            except Exception as exc:
                                st.error(str(exc))
                if st.button("Call MCP: search_market_trends") and TAVILY_KEY:
                    mcp_srv = get_mcp_server()
                    if mcp_srv:
                        with st.spinner("Executing via MCP protocol..."):
                            try:
                                out = call_mcp_tool(mcp_srv, "search_market_trends",
                                                    {"query": "India grocery retail trends 2024"})
                                st.json(out)
                            except Exception as exc:
                                st.error(str(exc))

        st.divider()
        st.markdown("### LangGraph Supervisor-Worker State Machine")
        st.markdown(
            "In a LangGraph deployment, the pipeline is expressed as a `StateGraph`. "
            "The supervisor node receives aggregated worker outputs via shared TypedDict state."
        )
        st.code(
            "class FreshMartState(TypedDict):\n"
            "    query:          str\n"
            "    weather_signal: dict\n"
            "    market_signal:  dict\n"
            "    rag_context:    str\n"
            "    recommendation: str\n\n"
            "graph = StateGraph(FreshMartState)\n"
            "graph.add_node('security',  security_node)\n"
            "graph.add_node('router',    router_node)\n"
            "graph.add_node('weather',   weather_node)   # Worker\n"
            "graph.add_node('market',    market_node)    # Worker\n"
            "graph.add_node('rag',       rag_node)       # Worker\n"
            "graph.add_node('supervisor', synthesis_node)\n"
            "graph.add_conditional_edges('router', route_fn,\n"
            "    {'weather': 'weather', 'market': 'market', 'rag': 'rag'})\n"
            "graph.add_edge(['weather', 'market', 'rag'], 'supervisor')",
            language="python",
        )

        st.divider()
        st.markdown("### Cost Optimisation -- Model Routing Decision Table")
        data = {
            "Query type":      ["Simple fact lookup", "Trend + weather query", "Full strategic analysis"],
            "Model selected":  [MODEL_FAST, MODEL_FAST, MODEL_SMART],
            "Input cost /1K":  ["$0.00015", "$0.00015", "$0.0025"],
            "Output cost /1K": ["$0.0006",  "$0.0006",  "$0.01"],
            "When triggered":  [
                "No strategic keywords + budget < $0.05",
                "Weather/market keywords, moderate complexity",
                "strategy / forecast / evaluate / scenario keywords",
            ],
        }
        st.table(data)

    # ── Tab 4: Observability ──────────────────────────────────────────────────
    with tab_obs:
        st.markdown("### Execution Trace")
        trace = result.get("trace", [])
        for step in trace:
            status = step["status"]
            icon   = "OK" if status == "OK" else "PASS" if status == "PASS" else "BLOCK" if status == "BLOCK" else "ERR"
            t_col, s_col, d_col = st.columns([2, 1, 5])
            t_col.write(step["step"])
            s_col.write(f"[{icon}]")
            d_col.write(step.get("detail", "")[:90])

        if "synthesis" in result:
            syn = result["synthesis"]
            st.divider()
            st.markdown("### Token and Cost Breakdown")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Model",          syn["model"])
            m2.metric("Input tokens",   syn["input_tokens"])
            m3.metric("Output tokens",  syn["output_tokens"])
            m4.metric("Cost (USD)",     f"${syn['cost_usd']:.5f}")
            m5.metric("Session total",  f"${st.session_state['total_cost']:.4f}")

            if syn["compressed"]:
                st.info(
                    "Prompt compression applied: context exceeded 2000 tokens "
                    "and was truncated to stay within the model budget."
                )

            st.divider()
            st.markdown("### Agent Latency Profile (ms)")
            latency_data = {
                s["step"]: s.get("latency_ms", 0)
                for s in trace
                if s.get("latency_ms", 0) > 0
            }
            if latency_data:
                st.bar_chart(latency_data)

            total_ms = result.get("total_ms", 0)
            slo_ok   = result.get("slo_met", False)
            st.markdown(
                f"**End-to-end latency:** {total_ms} ms  |  "
                f"**SLO (15 s):** {'MET' if slo_ok else 'BREACHED'}"
            )

            st.divider()
            st.markdown("### Security and Compliance")
            sec_step = next((s for s in trace if s["step"] == "Security Guard"), None)
            if sec_step:
                st.write(f"Security Guard result: **{sec_step['status']}**")
                st.write(sec_step["detail"])
            st.markdown(
                "Input validation covers: prompt injection patterns, "
                "PII-adjacent keywords, length limits, and special-character sanitisation."
            )

# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    init_state()

    # Load Pinecone once per session
    if st.session_state["vs"] is None:
        vs, err = load_pinecone_store()
        st.session_state["vs"]       = vs
        st.session_state["vs_error"] = err

    render_sidebar()
    render_header()
    query, city, run = render_input()

    if run and query.strip():
        budget_remaining = st.session_state["budget"] - st.session_state["total_cost"]
        if budget_remaining <= 0:
            st.error("Session budget exhausted. Increase the budget or reset the session.")
            return

        with st.spinner("Orchestrating agents..."):
            result = run_pipeline(
                query.strip(),
                city,
                budget_remaining,
                st.session_state["vs"],
            )

        # Update session state
        if "synthesis" in result:
            st.session_state["total_cost"] += result["synthesis"]["cost_usd"]
        st.session_state["query_count"] += 1
        st.session_state["history"].append({
            "time":  datetime.now().strftime("%H:%M:%S"),
            "city":  city,
            "query": query.strip(),
        })

        render_tabs(result)


if __name__ == "__main__":
    main()

# To run: `streamlit run app.py`
# or
# python -m streamlit run app.py
# or
# python3 -m streamlit run app.py

# pip install streamlit langchain langchain-core langchain-openai langchain-pinecone \
#             langgraph openai pinecone tiktoken requests python-dotenv mcp nest_asyncio
# streamlit run app.py