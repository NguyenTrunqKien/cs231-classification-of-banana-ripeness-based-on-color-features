import cv2
import numpy as np

def rotate_image_and_mask(image, mask, angle):
    """
    Xoay cả ảnh và mask theo một góc bất kỳ mà không làm thay đổi kích thước 224x224.
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Tạo ma trận xoay
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Thực hiện xoay ảnh (dùng INTER_LINEAR cho ảnh)
    rotated_img = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)

    # Thực hiện xoay mask (dùng INTER_NEAREST để giữ nguyên giá trị 0 và 255)
    rotated_mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)

    return rotated_img, rotated_mask

def add_gaussian_noise(image, sigma=15):
    """
    Thêm nhiễu Gaussian vào ảnh.
    - sigma: Độ lệch chuẩn của phân phối Gaussian
    """
    row, col, ch = image.shape
    mean = 0
    # Tạo nhiễu ngẫu nhiên
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    # Cộng nhiễu vào ảnh và ép kiểu về uint8
    noisy = np.clip(image.astype(np.float32) + gauss, 0, 255).astype(np.uint8)
    return noisy

# Hàm TĂNG CƯỜNG ẢNH BẰNG OPENCV (DATA AUGMENTATION)
def augment_image(img, mask):
    """
    Tạo ra các phiên bản biến đổi của ảnh gốc để tăng cường dữ liệu.
    """
    augmented_pairs = []
    # Ảnh gốc
    augmented_pairs.append((img, mask))

    # Thêm nhiễu
    img_noise = add_gaussian_noise(img)
    augmented_pairs.append((img_noise, mask))

    # Lật ngang (Horizontal Flip)
    img_flip_h = cv2.flip(img, 1)
    mask_flip_h = cv2.flip(mask, 1)
    augmented_pairs.append((img_flip_h, mask_flip_h))

    # Tăng độ sáng (Giả lập chụp dưới đèn mạnh)
    # Tăng thêm 30 đơn vị vào kênh V trong HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, 30)
    img_bright = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
    augmented_pairs.append((img_bright, mask))

    # Giảm độ sáng (Giả lập chụp thiếu sáng)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.subtract(v, 30)
    img_dark = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
    augmented_pairs.append((img_dark, mask))

    return augmented_pairs

#Với bài toán phân loại màu sắc, chúng ta nên ưu tiên tăng cường những thứ làm
#thay đổi giá trị Vector đặc trưng thay vì chỉ thay đổi vị trí điểm ảnh.
#Đề xuất 3 hướng chính:
#Thay đổi độ sáng (Brightness/Value): Rất quan trọng vì chuối chụp ở điều kiện thiếu sáng sẽ khác chuối chụp dưới đèn led.
#Lật ngang (Horizontal Flip): Để bù đắp sự bất đối xứng của camera, 1 lần là đủ.
#Nhiễu nhẹ (Gaussian Noise): Để mô hình không bị "sốc" khi ảnh bị nhiễu hạt.