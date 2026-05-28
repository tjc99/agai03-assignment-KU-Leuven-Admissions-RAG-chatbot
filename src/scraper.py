import os
import time
import requests
from bs4 import BeautifulSoup

# 鲁汶大学 Admissions Office 核心政策页面
URLS = {
    "apply_main": "https://www.kuleuven.be/english/apply",
    "how_to_apply": "https://www.kuleuven.be/english/apply/application-instructions/apply-to-kuleuven",
    "degree_instructions": "https://www.kuleuven.be/english/apply/application-instructions/instructions-degree",
    "requested_documents": "https://www.kuleuven.be/english/apply/requested-documents",
    "bachelor_eligibility": "https://www.kuleuven.be/english/apply/requested-documents/eligibility-bachelor",
    "process_details": "https://www.kuleuven.be/english/apply/application-instructions/application-details",
    "general_admission": "https://www.kuleuven.be/english/education/student/register/Admissionrequirements"
}

def scrape_ku_leuven():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    # 自动创建作业要求的 data/raw 目录 [cite: 66]
    os.makedirs("data/raw", exist_ok=True)
    
    for name, url in URLS.items():
        print(f"🔄 Fetching: {url} ...")
        try:
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                
                # 剥离多余导航、页脚（大作业硬性清洗要求 [cite: 30]）
                for nav in soup(["nav", "footer", "header", "script", "style", "aside"]):
                    nav.decompose()
                
                main_content = soup.find("div", id="content") or soup.find("main") or soup.body
                
                if main_content:
                    text = main_content.get_text(separator="\n")
                    cleaned_lines = [line.strip() for line in text.splitlines() if line.strip()]
                    final_text = "\n".join(cleaned_lines)
                    
                    # 写入 raw 文件夹 [cite: 66]
                    file_path = f"data/raw/{name}.txt"
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(f"Source URL: {url}\n\n" + final_text)
                    print(f"✅ Saved to: {file_path}")
                else:
                    print(f"❌ Cannot find content block for {name}")
            else:
                print(f"❌ Failed with status code: {response.status_code}")
                
            # 严格遵守 Robots.txt 道德合规：每爬一页延迟 2 秒 [cite: 22, 104]
            time.sleep(2)
            
        except Exception as e:
            print(f"💥 Exception on {name}: {str(e)}")

if __name__ == "__main__":
    scrape_ku_leuven()