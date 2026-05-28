


# 🎓 KU Leuven Admissions RAG Chatbot

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-LangChain-orange.svg)](https://python.langchain.com/)
[![Frontend](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io/)
[![VectorDB](https://img.shields.io/badge/VectorDB-ChromaDB-blueviolet.svg)](https://www.trychroma.com/)

An intelligent, production-ready **Retrieval-Augmented Generation (RAG)** chatbot engineered specifically for the **KU Leuven International Admissions Office**. 

This system implements a cutting-edge hybrid QA-vector retrieval pipeline, features a reactive Streamlit web interface, and undergoes strict rigorous validation via an automated **LLM-as-a-Judge (RAG Triad)** assessment pipeline.

> 📝 Developed as part of the **AGAI-03 Senior AI Assignment**.

---

## 🛠️ Tech Stack & Architecture

- **Orchestration Framework:** `LangChain Core`
- **Vector Database:** `ChromaDB` (Persistent Local Storage)
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (Executed locally)
- **LLM Core & Judge:** `Google Gemini 2.5 Flash` / `Gemini 2.5 Flash-Lite`
- **Frontend UI:** `Streamlit Framework`
- **Evaluation Engine:** Custom `RAG Triad` Matrix (Context Relevance & Faithfulness Metrics)

---

## 📂 Project Directory Structure

```text
web_RAG/
├── .env                         # Local environment configuration (Git ignored)
├── requirements.txt             # Project software dependencies
├── README.md                    # GitHub repository landing page
└── Data/
    ├── app.py                   # Streamlit Frontend UI & Core RAG Orchestrator
    ├── data/
    │   ├── raw/                 # Scraped official KU Leuven HTML raw contents
    │   └── processed/           # Generated qa_dataset.csv & evaluation_report.csv
    └── src/
        ├── scraper.py           # Ethical web scraping pipeline
        ├── build_vector_db.py   # Tokenization, Chunking & ChromaDB ingestion
        └── evaluate.py          # LLM-as-a-Judge automated evaluator script

```

---

## 🚀 Quick Start (Local Deployment)

Follow these precise operational steps to get your local admissions advisor up and running in less than 2 minutes.

### 1. Environment Setup & Installation

Ensure you have Python 3.10+ initialized. Activate your localized virtual environment and execute:

```bash
pip install -r requirements.txt

```

### 2. Configure Environment Variables (`.env`)

Create a `.env` file in your root workspace directory (`web_RAG/.env`) and populate it with your personal Gemini API Token:

```env
GEMINI_API_KEY=your_actual_gemini_api_key_here

```

### 3. Launch the Chatbot Application

Boot up the Streamlit engine from your terminal session:

```bash
streamlit run Data/app.py

```

Your default browser will automatically deploy the user interface at: `http://localhost:8501`

### 4. Run Automated Evaluation Pipeline

To replicate our academic evaluation matrix and audit system accuracy benchmarks:

```bash
python Data/src/evaluate.py

```

---

## 📊 Evaluation Matrix Summary

The system is routinely audited against production standards using the **RAG Triad** evaluation mechanism, tracking two vital KPIs:

* **Context Relevance:** Validates the semantic precision of the ChromaDB retrieval stage.
* **Faithfulness:** Enforces factual compliance and strict anti-hallucination guardrails.
