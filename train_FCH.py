import os
import numpy as np
from tqdm import tqdm
import joblib

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from src.utils import load_images_from_folder
from src.FCH_feature_extractor import extract_combined_features
from src.segmentation import process_banana_224
from src.augmentation import augment_image

# Đọc dữ liệu từ thư mục Dataset
base_dir = os.path.dirname(os.path.abspath(__file__))

train_folder = os.path.join(base_dir, "dataset", "Train")

train_images, train_labels, class_names = load_images_from_folder(train_folder)

print(f"Nhãn các class: {class_names}")
print(f"Tổng số ảnh trong tập train: {len(train_images)}")

# --- TẬP TRAIN: AUGMENT -> TRÍCH XUẤT ---
X_train = []
y_train = []

print("Đang thực hiện Augment và Trích xuất đặc trưng tập TRAIN...")

for idx, img in enumerate(tqdm(train_images)):
    # 1. Phân đoạn ảnh gốc để lấy mask chuẩn
    final_img, final_mask = process_banana_224(img)

    # Kiểm tra nếu phân đoạn thất bại
    if final_img is None or final_mask is None:
        print("Phân đoạn quả chuối thất bại")
        # Sử dụng ảnh đen và mask đen thay thế để các hàm sau không bị lỗi logic
        final_img = np.zeros((224, 224, 3), dtype=np.uint8)
        final_mask = np.zeros((224, 224), dtype=np.uint8)

    # Tạo danh sách ảnh tăng cường từ ảnh đã phân đoạn
    # Hàm augment_image trả về list các cặp [(img1, m1), (img2, m2),...]
    pairs = augment_image(final_img, final_mask)

    for aug_img, aug_mask in pairs:
        # Trích xuất đặc trưng cho từng bản augment
        feature = extract_combined_features(aug_img, aug_mask)

        if feature is not None:
            X_train.append(feature)
            # Mỗi bản augment đều mang nhãn của ảnh gốc
            y_train.append(train_labels[idx])

# Chuyển sang numpy array
X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"\n✅ HOÀN TẤT TRÍCH XUẤT")
print(f"Kích thước tập Train: {X_train.shape}, {y_train.shape}")

# Train model KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model_knn = KNeighborsClassifier(n_neighbors=5, metric='manhattan', weights='distance')
model_knn.fit(X_train_scaled, y_train)

# Lưu mô hình KNN và bộ chuẩn hóa Scaler vào file.
model_path , scaler_path ='models/knn_banana_model.pkl', 'models/scaler.pkl'
try:
    joblib.dump(model_knn, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"✅ Đã lưu mô hình thành công tại: {model_path}")
    print(f"✅ Đã lưu bộ chuẩn hóa thành công tại: {scaler_path}")
except Exception as e:
    print(f"❌ Lỗi khi lưu mô hình: {e}")