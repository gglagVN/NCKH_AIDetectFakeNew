import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, "Code", "Data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "Code", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

X = np.load(os.path.join(PROCESSED_DIR, "X.npy"))
y = np.load(os.path.join(PROCESSED_DIR, "y.npy"))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

print("🧠 Đang huấn luyện SVM...")
clf = SVC(C=10, kernel='rbf', probability=True, class_weight='balanced')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"📊 Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred))

joblib.dump(clf, os.path.join(MODEL_DIR, "svm_model.pkl"))
print("✅ Đã cập nhật svm_model.pkl")