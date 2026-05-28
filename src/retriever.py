import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. 載入環境變量與 API Key
project_root = "c:\\Users\\traje\\web_RAG\\Data"
load_dotenv(os.path.join("c:\\Users\\traje\\web_RAG", ".env"))
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

# 2. 初始化與昨天完全一致的本地 Embedding 模型和 ChromaDB
persist_db_dir = os.path.join(project_root, "chroma_db")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 載入本地向量資料庫
vector_store = Chroma(persist_directory=persist_db_dir, embedding_function=embeddings)

# 初始化 Gemini 大模型
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2, # 降低隨機性，保證回答嚴謹
    google_api_key=api_key
)

def ask_rag_bot(user_question):
    """
    RAG 核心問答鏈路
    """
    try:
        # 🎯 步驟 1：向量檢索 (從 Chroma 撈出最相關的 3 個文本塊)
        # similarity_search_with_score 會同時返回文檔和相似度得分
        docs_and_scores = vector_store.similarity_search_with_score(user_question, k=3)
        
        context_parts = []
        sources = set()
        
        for doc, score in docs_and_scores:
            context_parts.append(doc.page_content)
            # 提取昨天存入的元數據（元數據中的 source_page）
            if "source" in doc.metadata:
                sources.add(doc.metadata["source"])
        
        context_text = "\n\n".join(context_parts)
        
        # 🎯 步驟 2：建構強力的 Prompt 策略（防幻覺、限定範圍）
        system_prompt = (
            "You are an official admissions bot for KU Leuven. Your job is to answer the student's question "
            "accurately based ONLY on the provided context. If the answer cannot be found in the context, "
            "say 'I am sorry, but I cannot find that information in the official documentation. Please contact the admissions office.' "
            "Do not make up any facts.\n\n"
            f"Context from official website:\n{context_text}"
        )
        
        # 🎯 步驟 3：呼叫大模型生成答案
        response = llm.invoke([
            ("system", system_prompt),
            ("user", user_question)
        ])
        
        # 處理返回類型（兼容 List 和 Str）
        if isinstance(response.content, list):
            answer = "".join([part if isinstance(part, str) else part.get("text", "") for part in response.content])
        else:
            answer = str(response.content)
            
        return {
            "answer": answer.strip(),
            "sources": list(sources) if sources else ["Unknown"]
        }
        
    except Exception as e:
        return {
            "answer": f"❌ 系統檢索出錯: {str(e)}",
            "sources": []
        }

if __name__ == "__main__":
    # 測試本地 RAG 是否跑通
    test_query = "What are the English language requirements for Bachelor programs?"
    print(f"🤖 正在測試問答系統，問題: '{test_query}'...\n")
    result = ask_rag_bot(test_query)
    print(f"💡 AI 回答:\n{result['answer']}\n")
    print(f"📂 數據來源引用: {result['sources']}")