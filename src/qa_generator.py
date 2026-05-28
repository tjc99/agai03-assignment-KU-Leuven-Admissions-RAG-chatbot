import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. 載入環境變量
load_dotenv("c:\\Users\\traje\\web_RAG\\.env")

# 🎯 提取 API Key
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

# 🛠️ 模型選擇
MODEL_NAME = "gemini-2.5-flash" 

def generate_qa_pairs():
    print("==================================================")
    print("🎬 項目第二階段：Gemini 數據集生成器正式啟動！")
    print(f"🚀 當前選用的模型為: {MODEL_NAME}")
    print("==================================================")
    
    # 初始化 LangChain Google GenAI 客戶端
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME, 
        temperature=0.3,
        google_api_key=api_key
    )
    
    # 🎯 【硬核絕對路徑】直接鎖定硬碟物理位置，絕不允許任何玄學 Bug 發生
    raw_dir = "c:\\Users\\traje\\web_RAG\\Data\\data\\raw"
    processed_dir = "c:\\Users\\traje\\web_RAG\\Data\\data\\processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    print(f"🔍 正在檢查物理目錄: {raw_dir}")
    
    # 🚨 安全檢查：如果這個路徑在硬碟上根本不存在，立刻報警
    if not os.path.exists(raw_dir):
        print(f"❌ 嚴重錯誤：在你的電腦裡根本找不到 '{raw_dir}' 這個資料夾！請核對路徑拼寫。")
        return

    # 🚨 獲取檔案清單
    all_files = os.listdir(raw_dir)
    print(f"📂 該資料夾下目前存在的實體檔案有: {all_files}")
    
    txt_files = [f for f in all_files if f.endswith(".txt")]
    print(f"📚 其中被識別為待處理的 .txt 文本有: {txt_files}")
    
    if not txt_files:
        print("❌ 終止運行：因為待處理的 .txt 檔案清單為空，大模型無事可做。")
        return
        
    print(f"\n🔥 檢查通過！開始呼叫 Gemini 榨取 {len(txt_files)} 個文件的知識...")
    all_qa_data = []
    
    # 2. 讀取抓取的所有魯汶大學 txt 文件
    for file_name in txt_files:
        file_path = os.path.join(raw_dir, file_name)
        print(f"🤖 Gemini 正在讀取並生成 Q/A: {file_name} ...")
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        system_prompt = (
            "You are an expert academic admissions advisor for KU Leuven. "
            "Based on the provided website content, generate 30 highly practical and realistic "
            "Question-Answer pairs that prospective international students would actually ask.\n"
            "Requirements:\n"
            "1. Both questions and answers must be professional, accurate, and detailed based ONLY on the context.\n"
            "2. Output MUST be a valid JSON array of objects, with keys: 'question' and 'answer'.\n"
            "3. Do not include any conversational filler, intro, or markdown code blocks (like ```json). Just the raw JSON text.\n"
        )
        
        try:
            response = llm.invoke([
                ("system", system_prompt),
                ("user", f"Context content:\n{content}")
            ])
            
            content_obj = response.content
            if isinstance(content_obj, list):
                text_parts = []
                for part in content_obj:
                    if isinstance(part, str): text_parts.append(part)
                    elif isinstance(part, dict) and "text" in part: text_parts.append(part["text"])
                    elif hasattr(part, "text"): text_parts.append(getattr(part, "text"))
                raw_text = "".join(text_parts).strip()
            else:
                raw_text = str(content_obj).strip()
            
            if raw_text.startswith("```"):
                raw_text = raw_text.split("\n", 1)[1].rsplit("\n", 1)[0].strip()
                if raw_text.startswith("json"):
                    raw_text = raw_text.split("\n", 1)[1].strip()

            qa_list = json.loads(raw_text)
            
            for qa in qa_list:
                all_qa_data.append({
                    "question": qa.get("question"),
                    "answer": qa.get("answer"),
                    "source_page": file_name.replace(".txt", "")
                })
            print(f"✅ 成功從 {file_name} 中榨取出 {len(qa_list)} 個問答對！")
            
        except Exception as e:
            print(f"❌ 处理 {file_name} 时发生错误: {e}")
            traceback.print_exc()
            
    # 5. 匯出為大作業要求的 CSV 格式
    if all_qa_data:
        df = pd.DataFrame(all_qa_data)
        output_path = os.path.join(processed_dir, "qa_dataset.csv")
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\n🎉 恭喜！全套數據集建構成功！")
        print(f"📊 總計生成問答對數量: {len(df)} (作業目標: 100-200個)")
        print(f"💾 儲存路徑: {output_path}")
    else:
        print("\n❌ 未能成功生成任何數據。")

if __name__ == "__main__":
    generate_qa_pairs()