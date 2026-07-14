# 🚚 RotaLog AI — Intelligent Logistics Risk & Cost Optimization Dashboard

RotaLog AI is a next-generation, **100% local, privacy-centric AI decision-support system** tailored for road freight, fleet operations, and logistics risk management.

By unifying a Local Large Language Model (Aya LLM) and high-performance Vector Search (ChromaDB), RotaLog AI enables logistics directors to query historical fleet data seamlessly, map multi-stop routes in real-time, and generate production-ready operational optimization reports—**with zero token costs and absolute data privacy.**

---

## 🌟 Key Features

*   **⚡ State-Based Data Caching:** Dynamically processes and holds complex database filters in RAM (`st.session_state`), scaling up dashboard performance to 30x faster query execution.
*   **🧠 Local AI Risk & Cost Agent:** Leverages the multilingual **Aya LLM** via Ollama to benchmark incoming route requests against thousands of historical reference operations under strict corporate financial rules.
*   **📊 Integrated Performance Matrix:** Aggregates and plots real-time distribution metrics (Status, Technical Vehicle Inspection, Traffic Conditions, Weather Conditions, Vehicle Types) using a unified interactive Plotly matrix.
*   **🗺️ Live Route Mapping Engine:** Connects directly with open-source OSRM (Open Source Routing Machine) routing APIs to generate line-maps on Mapbox under dynamic highway scenarios.
*   **📥 Enterprise-Grade Reporting:** Formats generated AI analyses into highly styled executive briefs, offering single-click native PDF exports via `weasyprint` and `img2pdf`.
*   **💾 Smart Session Uploads:** Employs unique cryptographic identifiers to prevent duplicate file copies on the server while preserving full log archives.

---

## 🛠️ System Architecture & Data Flow

```
[ User Input / CSV Upload ] ──> [ Streamlit Frontend Layer ]
                                            │
         ┌──────────────────────────────────┴──────────────────────────────────┐
         ▼                                                                     ▼
[ Vector Ingestion Engine ]                                         [ Real-Time Route Sim ]
  ├─ sentence-transformers                                             └─ Open Source Routing Machine
  ├─ ChromaDB Vektör DB (Cosine Space)                                 └─ Plotly Line Mapbox
  └─ RAM State-Caching
         │
         ▼
[ Local LLM Layer (Ollama / Aya) ] ──> [ Structured System Prompts ] ──> [ PDF Export (Weasyprint) ]
```

---

## 🚀 Installation & Local Deployment

### 1. Prerequisites
Ensure you have Python 3.10+ installed and **Ollama** running locally on your hardware.

### 2. Pull the AI Model
Download the multilingual open-weights model using your terminal:
```bash
ollama pull aya
```

### 3. Clone and Setup Environment
```bash
git clone [https://github.com/YOUR_USERNAME/RotaLog-AI.git](https://github.com/YOUR_USERNAME/RotaLog-AI.git)
cd RotaLog-AI

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install system dependencies
pip install -r requirements.txt
```

### 4. Running the Application
Launch the intelligent decision panel instantly:
```bash
streamlit run app.py
```

---

## 🗃️ Repository File Structure

```
├── .streamlit/               # Streamlit styling configurations
├── CSV_DATA/                 # Server-side physical CSV audit archives
├── agent.py                  # AI Agent Layer, Vector DB triggers & Ollama generation
├── app.py                    # Core Frontend UI Layout, state-caches & dashboards
├── report_history.py         # JSON-based local report persistence layer
├── requirements.txt          # Third-party Python dependencies
└── README.md                 # Project technical documentation
```

---

## 🔒 Absolute Privacy & Financial Viability
Traditional enterprise AI pipelines rely heavily on third-party cloud APIs (OpenAI, Anthropic), introducing high per-token operating expenses and risky data compliance leakages for sensitive commercial logs.

**RotaLog AI completely breaks this paradigm:**
1.  **Zero API Overhead:** No token constraints, monthly tiers, or request limits.
2.  **Air-Gapped Compliance:** Fleet asset data, driver driving hours, and financial damage reports never leave your local physical bare-metal hardware.

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.
