import pandas as pd
import os
import trafilatura

# ================= ĐƯỜNG DẪN =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Đảm bảo lưu vào đúng Code/Data/raw
RAW_DATA_DIR = os.path.join(CURRENT_DIR, "Data", "raw")
os.makedirs(RAW_DATA_DIR, exist_ok=True)

def scrap_single_url(url):
    """Sử dụng trafilatura để lấy nội dung bài báo"""
    try:
        downloaded = trafilatura.fetch_url(url)
        # Trích xuất nội dung chính (loại bỏ quảng cáo, menu)
        content = trafilatura.extract(downloaded)
        
        if content and len(content) > 200:
            return content
        return None
    except Exception as e:
        print(f"Lỗi khi cào {url}: {e}")
        return None

def update_dataset(urls, label):
    file_name = "real_news.csv" if label == 0 else "fake_news.csv"
    file_path = os.path.join(RAW_DATA_DIR, file_name)
    
    new_data = []
    for url in urls:
        print(f"🔍 Đang cào: {url}")
        content = scrap_single_url(url)
        if content:
            new_data.append({"text": content, "label": label})
            print("✅ Thành công")
        else:
            print("❌ Không lấy được nội dung hoặc bài quá ngắn.")
    
    if new_data:
        df_new = pd.DataFrame(new_data)
        if os.path.exists(file_path):
            df_old = pd.read_csv(file_path)
            df_final = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_final = df_new
        
        df_final.to_csv(file_path, index=False)
        print(f"🚀 Đã cập nhật xong {len(new_data)} bài vào {file_name}")

if __name__ == "__main__":
    # Test với link tin thật
    tin_that = [
        "https://vnexpress.net/thu-tuong-pham-minh-chinh-hoi-dam-voi-tong-thong-duc-4835682.html",
        "https://vnexpress.net/gia-vang-nhan-tiep-tuc-tang-manh-4835700.html",
        "https://tuoitre.vn/tong-thong-duc-tham-viet-nam-mo-ra-chuong-moi-trong-hop-tac-20240123111545678.htm"
    ]
    
    if tin_that:
        update_dataset(tin_that, label=0)