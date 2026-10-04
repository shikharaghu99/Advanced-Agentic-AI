# Advanced Agentic AI

Hands-on notebooks, reference docs, a capstone, and an end-to-end project from the **Advanced Agentic AI Program (India Track)**. The material covers how to design, build, evaluate, secure, run and cost-optimise GenAI and agentic systems for production.

---

## Repository Structure

```
Advanced-Agentic-AI/
├── docs/                    # Reference material
│   ├── Advanced_Agentic_AI_Program_India_Track.pdf   # Program agenda
│   └── rag/                 # RAG concept docs + sample PDF used by RAG notebooks
├── notebooks/               # Course notebooks, numbered in learning order
├── capstone_project/        # Capstone problem statement + supporting docs
│   └── docs/                # ADR, ARB summary, deployment model, risk register, scorecard, rubric
└── projects/
    └── IN18_Project/        # FreshMart AI Category Command Center (Streamlit multi-agent app)
```

---

## Learning Path / Agenda

| # | Notebook | Topic |
|---|----------|-------|
| **Day 1 – Foundations** |||
| 01 | `01_IN01_AI_Architecture_Decision_Framework` | Choosing the right AI architecture |
| 02 | `02_RAG_System_LangChain_OpenAI` | Building a RAG pipeline with LangChain + OpenAI |
| **Day 2 – RAG, Integration & Agent Design** |||
| 03 | `03_RAG_Evaluation_Monitoring` | RAG evaluation with RAGAS and monitoring |
| 04 | `04_IN02_RAG_Prompting_FineTuning_Walmart` | RAG vs prompting vs fine-tuning |
| 05 | `05_IN03_API_MCP_Frameworks_Build_vs_Buy` | APIs, MCP, frameworks, build vs buy |
| 06a | `06a_IN04_Agent_Design_SingleAgent_vs_MultiAgent_Day2` | Single-agent vs multi-agent (Day 2 version) |
| **Day 3 – Orchestration, Tools, Reliability & Security** |||
| 06b | `06b_IN04_Agent_Design_SingleAgent_vs_MultiAgent_Day3` | Single-agent vs multi-agent (Day 3 version) |
| 07 | `07_IN05_Orchestration_Patterns_Sequential_Router_Supervisor` | Sequential, router and supervisor patterns |
| 08 | `08_IN06_Tool_Integration_A2A_Failure_Resilience` | Tool integration, A2A, failure resilience |
| 09 | `09_IN07_Production_Readiness_Reliability_Engineering` | Production readiness and reliability |
| 10 | `10_IN08_Security_GenAI_Systems` | Security for GenAI systems |
| **Day 4 – Governance, Evaluation & Observability** |||
| 11 | `11_IN09_Governance_Scaling_SLO_Resilience_Assessment` | Governance, scaling, SLOs |
| 12 | `12_IN10_Evaluation_Frameworks_RAG_Agent` | Evaluation frameworks for RAG and agents |
| 13 | `13_IN11_Benchmark_Regression_LLMJudge` | Benchmarks, regression tests, LLM-as-judge |
| 14 | `14_IN12_Logging_Tracing_Observability_Debugging` | Logging, tracing, observability |
| **Day 5 – Cost, FinOps & Architecture Review** |||
| 15 | `15_IN13_Token_Economics_Cost_Optimisation` | Token economics and cost optimisation |
| 16 | `16_IN14_FinOps_GenAI_Governance` | FinOps for GenAI |
| 17 | `17_IN15_Architecture_Review_Tradeoff_Documentation` | Architecture review and trade-off docs |
| **Capstone** |||
| IN16 | `capstone_project/IN16_Capstone_Problem_Statement` | End-to-end design exercise (see `capstone_project/docs/`) |
| IN18 | `projects/IN18_Project/app.py` | FreshMart AI – multi-agent Streamlit app (see its [README](projects/IN18_Project/README.md)) |

---

## Prerequisites

- **Python 3.10+**
- Jupyter (VS Code Jupyter extension, JupyterLab, or Notebook)
- API keys (put them in a `.env` file at the repo root and in `projects/IN18_Project/`):

```env
OPENAI_API_KEY=sk-...            # required by almost every notebook
PINECONE_API_KEY=pcsk_...        # vector DB (RAG notebooks / IN18 project)
TAVILY_API_KEY=tvly-...          # web search tool (IN18 project)
OPENWEATHERMAP_API_KEY=...       # weather tool (IN18 project)
```

> Never commit `.env`. Add it to `.gitignore`.

---

## Setup

```bash
git clone https://github.com/shikharaghu99/Advanced-Agentic-AI.git
cd Advanced-Agentic-AI

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -U pip
pip install jupyter ipykernel python-dotenv requests pandas tiktoken \
            openai langchain langchain-core langchain-openai langchain-community \
            langchain-text-splitters langchain-pinecone pinecone langgraph \
            ragas==0.4.3 pypdf mcp nest_asyncio streamlit
```

### Running the notebooks
1. Open a notebook from `notebooks/` and select the `.venv` kernel.
2. Run cells top to bottom, following the numbered order.
3. Run the RAG notebooks (02, 03) from inside `notebooks/` so the relative path `../docs/rag/company_policy.pdf` resolves.

### Running the IN18 project
```bash
cd projects/IN18_Project
streamlit run app.py
```
For the full walkthrough, see [projects/IN18_Project/README.md](projects/IN18_Project/README.md).

---

## Key Concepts Covered

- AI architecture decision-making: RAG vs prompting vs fine-tuning, build vs buy
- Agent design: single-agent vs multi-agent, orchestration patterns (LangGraph)
- Tool use, MCP, A2A, and failure resilience
- Production readiness, reliability engineering, and security
- Governance, SLOs, evaluation (RAGAS, LLM-as-judge), and regression benchmarks
- Observability: logging, tracing, debugging
- Token economics, cost optimisation, and FinOps
- Architecture review boards, ADRs, and trade-off documentation

---

## Acknowledgement

Content comes from the Advanced Agentic AI training program (Walmart India, July 2026) and is kept here for personal learning and reference.