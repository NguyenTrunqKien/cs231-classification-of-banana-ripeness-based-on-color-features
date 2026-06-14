import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.metrics import f1_score, classification_report, confusion_matrix

from src.utils import load_images_from_folder
from src.feature_extractor import compute_feature
from src.segmentation import process_banana_224

def evaluate(name, y_true, y_pred):
    print(f"===== {name} =====")
    print(f"F1 Macro: {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center')

    plt.colorbar()
    plt.show()


def evaluate_model(model_path='models/knn_banana_model.pkl', scaler_path='models/scaler.pkl'):
    # Load model và scaler đã lưu
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Đọc dữ liệu từ thư mục Dataset
    base_dir = os.path.dirname(os.path.abspath(__file__))

    test_folder  = os.path.join(base_dir, "dataset", "Test")

    test_images, test_labels, _ = load_images_from_folder(test_folder)

    print(f"Tổng số ảnh trong tập test: {len(test_images)}")

    # --- TẬP TEST: TRÍCH XUẤT (KHÔNG AUGMENT) ---
    X_test = []
    y_test = []

    print("\nĐang trích xuất đặc trưng tập TEST...")
    for idx, img in enumerate(tqdm(test_images)):
        # 1. Phân đoạn ảnh gốc để lấy mask chuẩn
        final_img, final_mask = process_banana_224(img)

    # Kiểm tra nếu phân đoạn thất bại
    if final_img is None or final_mask is None:
        print("Phân đoạn quả chuối thất bại")
        # Sử dụng ảnh đen và mask đen thay thế để các hàm sau không bị lỗi logic
        final_img = np.zeros((224, 224, 3), dtype=np.uint8)
        final_mask = np.zeros((224, 224), dtype=np.uint8)

    # Trích xuất đặc trưng
    feature = compute_feature(final_img, final_mask)

    if feature is not None:
        X_test.append(feature)
        y_test.append(test_labels[idx])

    # Chuyển sang numpy array
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    print(f"\n✅ HOÀN TẤT TRÍCH XUẤT")
    print(f"Kích thước tập Test:  {X_test.shape}, {y_test.shape}")

    # Kiểm thử trên tập Test
    X_test_scaled = scaler.transform(X_test)
    y_test_pred = model.predict(X_test_scaled)

    print("\n📊 BÁO CÁO CHI TIẾT TRÊN TẬP KIỂM TRA (TEST SET):")
    evaluate("Test", y_test, y_test_pred)

evaluate_model()