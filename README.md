# NCKH: Nghiên cứu và phát triển AI phát hiện tin giả trên mạng xã hội

<p align=\"center\">
  <img src=\"https://thuvienvector.vn/wp-content/uploads/2025/09/logo-haui.jpg\" alt=\"Trường Đại học Công nghiệp Hà Nội\" width=\"220\"/>
</p>

## Tổng quan
Đề tài nghiên cứu ứng dụng các kỹ thuật xử lý ngôn ngữ tự nhiên (NLP) và học sâu (Deep Learning) để phát hiện tin giả tiếng Việt trên các nền tảng mạng xã hội. Mục tiêu xây dựng mô hình phân loại văn bản (tin thật / tin giả) và một ứng dụng demo để minh họa kết quả.

## Mục tiêu
- Xây dựng mô hình AI sử dụng NLP để phân loại tin thật và tin giả trên mạng xã hội.
- Nâng cao độ chính xác cho tiếng Việt bằng tối ưu hóa và tinh chỉnh các mô hình như PhoBERT, BERT, LSTM.
- Cung cấp bộ dữ liệu và ứng dụng demo hỗ trợ kiểm chứng thông tin.

## Cách hoạt động (tóm tắt quy trình)
1. Thu thập dữ liệu:
   - Thu thập văn bản (bài báo, bài đăng mạng xã hội, dataset công khai).
   - Gán nhãn tin thật / tin giả bằng phương pháp bán tự động và kiểm duyệt thủ công.

2. Tiền xử lý:
   - Chuẩn hóa tiếng Việt (unicode, dấu câu).
   - Tokenization phù hợp tiếng Việt (sử dụng các tokenizer cho PhoBERT / vnCorpora).
   - Loại bỏ stopwords, xử lý từ viết tắt, tiền xử lý URL/hashtag/mention khi cần.

3. Huấn luyện mô hình:
   - Thử nghiệm các mô hình cơ bản: Logistic Regression, SVM, LSTM.
   - Thử nghiệm các mô hình tiền huấn luyện ngôn ngữ: PhoBERT, BERT Vietnamese, tinh chỉnh (fine-tune).
   - So sánh, chọn mô hình tốt nhất theo Accuracy, Precision, Recall, F1-score.

4. Đánh giá và tinh chỉnh:
   - Đánh giá trên tập validation/test độc lập.
   - Điều chỉnh hyperparameters, làm cân bằng dữ liệu nếu cần (oversampling/undersampling).

5. Triển khai demo:
   - Xây dựng ứng dụng web đơn giản (Flask/FastAPI) cho phép người dùng nhập văn bản và nhận nhãn/tỉ lệ tin giả.
   - Hiển thị các chỉ số giải thích (ví dụ: attention/độ tin cậy) nếu mô hình hỗ trợ.

## Dữ liệu
- Dự án sử dụng dữ liệu tiếng Việt thu thập từ nhiều nguồn (báo chí, mạng xã hội, dataset công khai).
- Tất cả dữ liệu nhạy cảm sẽ được xử lý tuân thủ quy định và bản quyền nguồn cung cấp.

## Hướng dẫn chạy nhanh (demo)
1. Cài môi trường:
   - Python 3.8+; khuyến nghị tạo virtualenv.
   - Cài dependencies: `pip install -r requirements.txt`

2. Chuẩn bị dữ liệu / mô hình:
   - Đặt file dataset vào thư mục `data/`.
   - Nếu có mô hình tiền huấn luyện, đặt file checkpoint vào `models/`.

3. Chạy demo (ví dụ Flask):
   - `export FLASK_APP=app.py`
   - `flask run --host=0.0.0.0 --port=5000`
   - Mở trình duyệt đến http://localhost:5000 để thử nghiệm.

## Kết quả mong đợi
- Bộ dữ liệu tiếng Việt cho phát hiện tin giả.
- Mô hình phân loại tin giả/tin thật với độ chính xác mục tiêu >85%.
- Ứng dụng demo để trình diễn và đánh giá.

## Nhóm thực hiện và liên hệ
- Chủ nhiệm đề tài: Đặng Tuấn Linh  
  - Mã số sinh viên: 2023601070  
  - Lớp: 2023DHKTPM - K18  
  - Điện thoại: 0911 562 129  
  - Email: gglag2112005@gmail.com

- Giảng viên hướng dẫn: ThS. Nguyễn Tuấn Tú  
  - Đơn vị: Khoa Công nghệ thông tin, Trường Đại học Công nghiệp Hà Nội  
  - Điện thoại: 0968 581 566  
  - Email: Tunt@haui.edu.vn

- Sinh viên tham gia:
  | STT | Họ và tên | Mã số sinh viên | Lớp |
  |---|---|---|---|
  | 1 | Đặng Tuấn Linh | 2023601070 | 2023DHKTPM01-K18 |
  | 2 | Đinh Minh Cảnh | 2023601261 | 2023DHKTPM01-K18 |
  | 3 | Bùi Thanh Bình | 2023600790 | 2023DHKTPM01-K18 |
  | 4 | Nguyễn Đăng Khôi | 2023600817 | 2023DHKTPM01-K18 |

## Giấy tờ & Cam kết
- Tên, nội dung và kết quả nghiên cứu không trùng lặp với các công trình đã công bố; đề tài không phải đồ án/khóa luận tốt nghiệp.
- Nhóm và giảng viên hướng dẫn chịu trách nhiệm nếu không thực hiện đúng kế hoạch theo quy định nhà trường.

---