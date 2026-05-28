# agai03-assignment-KU-Leuven-Admissions-RAG-chatbot

# KU Leuven Admissions RAG Chatbot

An intelligent, production-ready Retrieval-Augmented Generation (RAG) chatbot developed for the **KU Leuven International Admissions Office**. This project implements a cutting-edge hybrid QA-vector retrieval pipeline, features a user-friendly Streamlit web interface, and undergoes strict automated evaluation via an LLM-as-a-Judge pipeline.

Developed as part of the **AGAI-03 Senior AI Assignment**.

---

## 🚀 Quick Start (Local Deployment)

Follow these precise steps to get your local admissions advisor up and running in less than 2 minutes.

### 1. Environment Setup & Installation
Ensure you have Python 3.10+ installed. Activate your virtual environment and run:
```bash
pip install -r requirements.txt

### 2. Configure Environment Variables(.env)
Create a .env file in the root directory (web_RAG/.env) and populate it with your Gemini API Key:
GEMINI_API_KEY=your_actual_gemini_api_key_here

3. Launch the Chatbot Application
Run the Streamlit web application from your terminal:
streamlit run Data/app.py
Your browser will automatically open the UI at http://localhost:8501.

4. Run Automated Evaluation Pipeline
To replicate our academic evaluation matrix and view accuracy scores:
python Data/src/evaluate.py

Project Directory Structure
web_RAG/
│
├── .env                         # Local environment configuration (Git ignored)
├── requirements.txt             # Project dependencies
├── README.md                    # GitHub landing page & guide
│
└── Data/
    ├── app.py                   # Streamlit Frontend UI & Core RAG Orchestrator
    │
    ├── data/
    │   ├── raw/                 # Scraped official KU Leuven HTML contents
    │   └── processed/           # Generated qa_dataset.csv & evaluation_report.csv
    │
    └── src/
        ├── scraper.py           # Ethical web scraping pipeline
        ├── build_vector_db.py   # Tokenization, Chunking & ChromaDB ingestion
        └── evaluate.py          # LLM-as-a-Judge RAG Triad automated evaluator

🛠️ Tech Stack & Architecture
Orchestration: LangChain Core

Vector Database: ChromaDB (Persistent Local Storage)

Embeddings: sentence-transformers/all-MiniLM-L6-v2 (Local execution)

LLM Core & Judge: Google Gemini 2.5 Flash / Gemini 2.5 Flash-Lite

Frontend UI: Streamlit Framework

Evaluation System: Custom RAG Triad (Context Relevance & Faithfulness Metrics)



