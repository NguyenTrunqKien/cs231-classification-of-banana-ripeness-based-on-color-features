import numpy as np
import cv2
import joblib
import matplotlib.pyplot as plt

from src.feature_extractor import compute_feature
from src.segmentation import process_banana_224
from src.utils import show_image

def predict_new_image(image_path, model, scaler, class_names):
    # 1. Đọc ảnh mới
    raw_img = cv2.imread(image_path)
    if raw_img is None:
        print("Không đọc được ảnh!")
        return

    # 2. Phân đoạn để lấy vùng chứa quả chuối (tránh dùng biến global)
    # manual_mode=False để YOLO tự chạy
    final_img, final_mask = process_banana_224(raw_img)

    # Hiển thị ảnh final_img và final_mask
    show_image(final_img)
    show_image(final_mask)

    # Kiểm tra nếu phân đoạn thất bại
    if final_img is None or final_mask is None:
        print("Phân đoạn quả chuối thất bại")
        # Sử dụng ảnh đen và mask đen thay thế để các hàm sau không bị lỗi logic
        final_img = np.zeros((224, 224, 3), dtype=np.uint8)
        final_mask = np.zeros((224, 224), dtype=np.uint8)

    # Trích xuất đặc trưng
    feature = compute_feature(final_img, final_mask)

    if not feature:
        print("Lỗi trích xuất đặc trưng!")
        return

    # 4. Lấy vector duy nhất đó và chuẩn hóa
    current_feat = np.array(feature).reshape(1, -1)
    feat_scaled = scaler.transform(current_feat)

    # 5. Dự đoán
    pred_idx = model.predict(feat_scaled)[0]
    prob = model.predict_proba(feat_scaled)[0]

    label = class_names[pred_idx]
    confidence = prob[pred_idx] * 100

    # 6. Hiển thị (Dùng trực tiếp p_img vừa xử lý)
    plt.figure(figsize=(8, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB))
    plt.title("Ảnh gốc")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Dự đoán: {label} ({confidence:.2f}%)")
    plt.show()

    print(f"--- KẾT QUẢ CUỐI CÙNG: {label} ---")

model_path, scaler_path='models/knn_banana_model.pkl', 'models/scaler.pkl'
my_model = joblib.load(model_path)
my_scaler = joblib.load(scaler_path)
class_names = ['Overripe', 'Ripe', 'Unripe']

predict_new_image("/content/dd.png", my_model, my_scaler, class_names)