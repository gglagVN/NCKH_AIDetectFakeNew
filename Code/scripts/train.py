import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# 1. Thiết lập đường dẫn
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, "Code", "Data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "Code", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# 2. Tải dữ liệu
X = np.load(os.path.join(PROCESSED_DIR, "X.npy"))
y = np.load(os.path.join(PROCESSED_DIR, "y.npy"))

# Tách tập train/test (giữ 15% để test để tăng dữ liệu học)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

# 3. Huấn luyện SVM (Tham số C=10 giúp học kỹ hơn)
print("🧠 Đang huấn luyện mô hình SVM tối ưu...")
clf = SVC(C=10, kernel='rbf', probability=True, class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)

# 4. Đánh giá
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100
print(f"\n📊 KẾT QUẢ: {acc:.2f}%")
print(classification_report(y_test, y_pred))

# 5. Lưu mô hình
joblib.dump(clf, os.path.join(MODEL_DIR, "svm_model.pkl"))
print("✅ Đã lưu mô hình mới tại Code/models/svm_model.pkl")