import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


weights = models.EfficientNet_B0_Weights.DEFAULT
model_dl = models.efficientnet_b0(weights=weights)
model_dl.classifier = torch.nn.Identity() 
model_dl.eval() # chuyển sang suy luận không train

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # từ dataset ImageNet
])

def compute_fuzzy_hist(pixels, min_val, max_val, num_bins, is_circular=False):
    if len(pixels) == 0:
        return np.zeros(num_bins, dtype=np.float32)
        
    # Tính kích thước mỗi bin và xác định điểm tâm của từng bin
    bin_width = float(max_val - min_val) / num_bins
    bin_centers = np.linspace(min_val + bin_width/2, max_val - bin_width/2, num_bins) # np.linspace(start, stop, num), lấy ra tâm các bin
    
    fuzzy_hist = np.zeros(num_bins, dtype=np.float32) # mảng rỗng chứa kết quả bin
    
    for i, center in enumerate(bin_centers):
        # Tính khoảng cách từ mỗi pixel đến tâm của bin hiện tại
        dist = np.abs(pixels - center)
        
        # Nếu là kênh tuần hoàn (như kênh Hue 0-180), khoảng cách từ 179 đến 0 là 1 chứ không phải 179 vì kênh màu tuần hoàn 179 gần 0 chứ không phải cách xa nhau
        if is_circular:
            dist = np.minimum(dist, max_val - dist)
            
        # Gán trọng số: Pixel càng gần tâm bin thì trọng số càng gần 1. Vượt ra khỏi bin_width thì bằng 0.
        weights = 1.0 - (dist / bin_width)
        weights[weights < 0] = 0 # <0 nghĩa là khoảng cách rất xa với bin đang xét
        
        # Cộng dồn trọng số vào bin thay vì đếm số lượng (+1) như histogram cứng thông thường
        fuzzy_hist[i] = np.sum(weights)
        
    # Chuẩn hóa (Normalize) histogram để tổng các bin luôn bằng 1, giúp bất biến với kích thước ảnh chuối to/nhỏ
    total_weight = np.sum(fuzzy_hist)
    if total_weight > 0:
        fuzzy_hist /= total_weight
        
    return fuzzy_hist

def extract_color_features(img, mask, num_bins=16, is_rgb_input=True):
    # 1. Chuyển đổi hệ màu (Hỗ trợ cả đầu vào từ Web và Local)
    if is_rgb_input:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    else:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
    features = []

    # 2. Tạo boolean mask (Chỉ lấy các pixel có giá trị 255 - thuộc về quả chuối)
    # Bước này thay thế cho clean_mask cũ
    bool_mask = (mask == 255)
    
    # Kiểm tra an toàn: Nếu mask trống (không tìm thấy chuối)
    total_pixels = np.sum(bool_mask) 
    if total_pixels == 0: 
        return np.zeros(num_bins * 4, dtype=np.float32) 

    # 3. Lấy dữ liệu pixel hợp lệ dựa trên bool_mask
    h_channel = hsv[:,:,0][bool_mask].astype(np.float32)
    s_channel = hsv[:,:,1][bool_mask].astype(np.float32)
    a_channel = lab[:,:,1][bool_mask].astype(np.float32)
    b_channel = lab[:,:,2][bool_mask].astype(np.float32)

    # 4. Trích xuất Fuzzy Histogram (Giữ nguyên cấu trúc mượt mà)
    features.extend(compute_fuzzy_hist(h_channel, 0, 180, num_bins, is_circular=True))
    features.extend(compute_fuzzy_hist(s_channel, 0, 255, num_bins, is_circular=False))
    
    features.extend(compute_fuzzy_hist(a_channel, 0, 255, num_bins, is_circular=False))
    features.extend(compute_fuzzy_hist(b_channel, 0, 255, num_bins, is_circular=False))

    return np.array(features, dtype=np.float32)

def extract_feature_dl(masked_img):
    # Ảnh truyền vào đây đã được xóa nền ở hàm gộp, không cần tham số mask nữa
    img_rgb = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    input_tensor = preprocess(img_pil).unsqueeze(0) 
    
    with torch.no_grad():
        features = model_dl(input_tensor)
        
    return features.squeeze().numpy().astype(np.float32)

def extract_combined_features(cropped_img, mask, use_color=True, use_dl=False):
    combined_features = []

    masked_img = cv2.bitwise_and(cropped_img, cropped_img, mask=mask)

    if use_color:
        color_features = extract_color_features(
            cropped_img, 
            mask, 
            num_bins=16,     
            is_rgb_input= False # Đặt False nếu ảnh đầu vào đang là BGR
        )
        combined_features.append(color_features)
    
    if use_dl:
        dl_features = extract_feature_dl(masked_img)
        combined_features.append(dl_features)
        
    if len(combined_features) > 0:
        return np.concatenate(combined_features)
    else:
        return np.array([], dtype=np.float32)