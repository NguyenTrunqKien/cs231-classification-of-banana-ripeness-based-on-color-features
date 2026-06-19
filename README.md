# 🍌 Phân Loại Độ Chín Của Chuối Dựa Trên Đặc Trưng Màu Sắc

> Dự án xây dựng và so sánh hai phương pháp phân loại độ chín của chuối dựa trên đặc trưng màu sắc: **DCM (Dominant and Color Moments)** và **FCH (Fuzzy Color Histogram)**, kết hợp với bộ phân loại KNN.

---

## 🔬 Tổng Quan Dự Án

Mục tiêu của project là phân loại mức độ chín của chuối thông qua đặc trưng màu sắc trích xuất từ ảnh, sử dụng hai phương pháp biểu diễn màu khác nhau và đánh giá hiệu năng của từng phương pháp.

---

## 📁 Cấu Trúc Thư Mục

```
project/
├── figures/                        # Ảnh minh họa, biểu đồ kết quả
├── models/
│   ├── DCM/                        # Model và scaler đã huấn luyện (DCM)
│   │   ├── banana_features.py
│   │   ├── knn_banana_model.pkl
│   │   └── scaler.pkl
│   └── knn_fuzzy_color_histogram/  # Model đã huấn luyện (FCH)
├── notebooks/
│   ├── Demo_Classify.ipynb         # Demo phân loại ảnh chuối trực quan
│   ├── Train_DCM.ipynb             # Huấn luyện và đánh giá mô hình DCM
│   └── Train_FCH_DL.ipynb          # Huấn luyện và đánh giá mô hình FCH
├── src/
│   ├── DCM_feature_extractor.py    # Trích xuất đặc trưng theo phương pháp DCM
│   ├── FCH_feature_extractor.py    # Trích xuất đặc trưng theo phương pháp FCH
│   ├── augmentation.py             # Tăng cường dữ liệu ảnh
│   ├── segmentation.py             # Phân đoạn vùng chuối trong ảnh
│   ├── utils.py                    # Các hàm tiện ích dùng chung
│   └── __init__.py
├── evaluate.py                     # Đánh giá mô hình trên tập test
├── predict.py                      # Dự đoán độ chín từ ảnh đầu vào
├── train_DCM.py                    # Script huấn luyện mô hình DCM
├── train_FCH.py                    # Script huấn luyện mô hình FCH
└── requirements.txt                # Các thư viện cần thiết
```

---

## ⚙️ Yêu Cầu Hệ Thống

- Python **3.8+**
- pip hoặc conda
- Jupyter Notebook / JupyterLab

---

## 🚀 Hướng Dẫn Cài Đặt và Chạy Từ A đến Z

### Bước 1 — Clone hoặc Tải Về Project

```bash
# Nếu dùng Git
git clone <repository_url>
cd <tên_thư_mục>

# Hoặc giải nén file ZIP đã tải về
```

### Bước 2 — Tạo Môi Trường Ảo (Khuyến Nghị)

```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo
# Trên Windows:
venv\Scripts\activate

# Trên macOS/Linux:
source venv/bin/activate
```

### Bước 3 — Cài Đặt Các Thư Viện

```bash
pip install -r requirements.txt
```

---

## 🏋️ Huấn Luyện Mô Hình

Hai phương pháp DCM và FCH có quy trình huấn luyện hoàn toàn giống nhau, chỉ khác ở cách trích xuất đặc trưng màu sắc. Mỗi script huấn luyện sẽ tự động gọi các hàm xử lý được định nghĩa sẵn trong thư mục `src/`.

### Huấn luyện với phương pháp DCM

```bash
python train_DCM.py
```

Quá trình này sẽ gọi tới `src/DCM_feature_extractor.py`, `src/segmentation.py`, `src/augmentation.py` và lưu model vào `models/DCM/`.

### Huấn luyện với phương pháp FCH

```bash
python train_FCH.py
```

Quá trình này sẽ gọi tới `src/FCH_feature_extractor.py`, `src/segmentation.py`, `src/augmentation.py` và lưu model vào `models/knn_fuzzy_color_histogram/`.

---

## 📊 Đánh Giá và Dự Đoán

### Đánh giá mô hình trên tập test

```bash
python evaluate.py
```

### Dự đoán độ chín từ ảnh mới

```bash
python predict.py --image <đường_dẫn_tới_ảnh>
```

---

## 📓 Notebooks — Xem Trực Quan Kết Quả

> ⚠️ **Lưu ý:** Các notebook dưới đây là nơi thể hiện **đầy đủ và trực quan nhất** toàn bộ quá trình thực nghiệm, bao gồm ảnh minh họa, biểu đồ đánh giá, confusion matrix và phân tích kết quả. Vui lòng mở các notebook này để xem kết quả chi tiết của từng phương pháp.

| Notebook | Mô tả |
|:---|:---|
| `Train_DCM.ipynb` | Toàn bộ pipeline huấn luyện, đánh giá và trực quan hóa kết quả của phương pháp **DCM** |
| `Train_FCH_DL.ipynb` | Toàn bộ pipeline huấn luyện, đánh giá và trực quan hóa kết quả của phương pháp **FCH** |
| `Demo_Classify.ipynb` | Demo phân loại trực tiếp trên ảnh chuối mới, hiển thị kết quả dự đoán kèm hình ảnh |

### Khởi Động Jupyter Notebook

```bash
jupyter notebook
```

Trình duyệt sẽ tự động mở tại `http://localhost:8888`. Điều hướng vào thư mục `notebooks/` và mở lần lượt từng file.

---

## 🔄 Tóm Tắt Luồng Xử Lý

```
Ảnh đầu vào
      ↓
Phân đoạn vùng chuối     →  src/segmentation.py
      ↓
Tăng cường dữ liệu       →  src/augmentation.py
      ↓
Trích xuất đặc trưng
  ├── DCM                 →  src/DCM_feature_extractor.py   →  train_DCM.py
  └── FCH                 →  src/FCH_feature_extractor.py   →  train_FCH.py
      ↓
Huấn luyện KNN           →  models/
      ↓
Đánh giá & Dự đoán       →  evaluate.py / predict.py
```

---

## 👥 Thành Viên Nhóm

Nguyễn Trung Kiên - 24520888@gm.uit.edu.vn

Trần Gia Phú - 24521365@gm.uit.edu.vn

---

Dưới sự hướng dẫn của TS. Mai Tiến Dũng.

## 📄 Giấy Phép

Dự án được phát triển phục vụ mục đích học thuật trong khuôn khổ môn học **CS231 — Đại học Công nghệ Thông tin, ĐHQG TP.HCM**.
