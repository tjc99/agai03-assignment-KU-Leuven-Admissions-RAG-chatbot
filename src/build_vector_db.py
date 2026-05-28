import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def build_vector_db():
    print("==================================================")
    print("🎬 項目第三階段：Chroma 向量數據庫入庫程式啟動！")
    print("==================================================")

    # 🎯【硬核絕對路徑】直接精準鎖定你的物理硬碟位置
    project_root = "c:\\Users\\traje\\web_RAG\\Data"
    raw_dir = os.path.join(project_root, "data", "raw")
    persist_db_dir = os.path.join(project_root, "chroma_db")  # 向量數據庫本地持久化目錄
    
    # 1. 檢查原始文件
    if not os.path.exists(raw_dir) or not os.listdir(raw_dir):
        print(f"❌ 錯誤：未在資料夾 '{raw_dir}' 中找到原始爬取文本，請確認爬蟲。")
        return
        
    documents = []
    metadatas = []
    
    print(f"📂 正在從以下目錄讀取網頁文本: {raw_dir}")
    txt_files = [f for f in os.listdir(raw_dir) if f.endswith(".txt")]
    print(f"📄 偵測到待切片的文本文件: {txt_files}")
    
    # 2. 文本切片（大作業推薦的高級策略：RecursiveCharacterTextSplitter）
    # chunk_size=600字，overlap=60字，保證語意連續、不斷層（報告加分點）
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
    
    for file_name in txt_files:
        file_path = os.path.join(raw_dir, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            text_content = f.read()
            
        chunks = text_splitter.split_text(text_content)
        
        for chunk in chunks:
            documents.append(chunk)
            # 💡 記錄元數據元數據（source）：方便後續 Streamlit 介面做“數據源引用展示”（作業硬性要求）
            metadatas.append({"source": file_name.replace(".txt", "")})
            
    print(f"✂️ 文本切片完成！總共將原始長文章切分成了 {len(documents)} 個語意文本塊。")
    
    # 3. 初始化開源 Embedding 模型
    # 我們採用 sentence-transformers 的 all-MiniLM-L6-v2。完全免費、本地運行、不需要任何 Key！
    print("\n🤖 正在加載開源 Embedding 向量化模型 (sentence-transformers/all-MiniLM-L6-v2)...")
    print("💡 提示：如果是第一次運行，系統會自動從 HuggingFace 下載模型（約 90MB），請保持網路暢通。")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 4. 創建並持久化 Chroma 數據庫
    print("\n🔮 正在將文本塊轉換為向量並寫入 Chroma 數據庫（這可能需要 30 秒 - 1 分鐘）...")
    vector_store = Chroma.from_texts(
        texts=documents,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=persist_db_dir
    )
    
    print("\n==================================================")
    print(f"🎉 向量數據庫構建成功！")
    print(f"💾 數據庫已安全持久化存儲至本地目錄: `{persist_db_dir}`")
    print("==================================================")

if __name__ == "__main__":
    build_vector_db()