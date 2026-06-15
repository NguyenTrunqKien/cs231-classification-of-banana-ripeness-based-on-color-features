import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import skew


def preprocess_image(img, mask, filter_type='gaussian'):
    """
    Nhận trực tiếp final_img và final_mask từ YOLO để tối ưu hiệu suất.
    """
    if filter_type == 'gaussian':
        img_filtered = cv2.GaussianBlur(img, (5, 5), 0)
    else:
        img_filtered = cv2.medianBlur(img, 5)

    img_hsv = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)

    # Dùng mask để chỉ tính Histogram (CLAHE) trên phần quả chuối, tránh nhiễu nền đen
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_eq = clahe.apply(v)

    img_hsv_eq = cv2.merge((h, s, v_eq))

    # Output ảnh HSV và mask
    return img_hsv_eq, mask

def extract_color_features(img_hsv, mask, k_clusters=3, w1=0.3, w2=0.4, w3=0.3):
    """
    Sử dụng K-means để tìm cụm màu và tính toán các đặc trưng Ck, Ak, Sk, pk.
    """
    banana_pixels = img_hsv[mask == 255]
    if len(banana_pixels) == 0:
        return None # Tránh lỗi nếu mask trống

    pixel_data = np.float32(banana_pixels)

    # K-Means phân cụm màu
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixel_data)
    centers = kmeans.cluster_centers_

    v_channel = img_hsv[:, :, 2]
    sobelx = cv2.Sobel(v_channel, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(v_channel, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    gradient_pixels = gradient_magnitude[mask == 255]

    _, counts = np.unique(labels, return_counts=True)
    max_area = np.max(counts)

    green_ratio, yellow_ratio, brown_ratio = 0.0, 0.0, 0.0
    pk_list = []

    for i in range(k_clusters):
        cluster_mask = (labels == i)
        Ak = counts[i] / max_area
        Sk = centers[i][1] / 255.0
        Ck = np.mean(gradient_pixels[cluster_mask])
        Ck_normalized = min(Ck / 255.0, 1.0)

        pk = w1 * Ck_normalized + w2 * Ak + w3 * Sk
        pk_list.append(pk)

        hue = centers[i][0]
        area_ratio = counts[i] / len(banana_pixels)
        if 35 <= hue <= 85: green_ratio += area_ratio
        elif 20 < hue < 35: yellow_ratio += area_ratio
        else: brown_ratio += area_ratio

    mean_pk = np.mean(pk_list)

    moments = []
    for i in range(3):
        channel_pixels = banana_pixels[:, i]
        mean_val = np.mean(channel_pixels) / 255.0
        std_val = np.std(channel_pixels) / 255.0
        skew_val = skew(channel_pixels)
        moments.extend([mean_val, std_val, skew_val])

    feature_vector = [yellow_ratio, green_ratio, brown_ratio, mean_pk] + moments
    return feature_vector

# Hàm trích xuất đặc trưng
def compute_feature(img, mask):
    """
    Hàm trích xuất đặc trưng màu sắc.
    """
    # Tiền xử lý để lấy ảnh HSV đã được cân bằng sáng
    img_hsv, processed_mask = preprocess_image(img, mask)

    features = extract_color_features(img_hsv, processed_mask)

    return features