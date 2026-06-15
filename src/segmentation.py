import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

model_yolo = YOLO('yolov8x-seg.pt')

def process_banana_224(img, img_idx=None, debug=False):
    h_orig, w_orig = img.shape[:2]

    results = model_yolo.predict(source=img, verbose=False, classes=[46])
    result = results[0]

    if result.masks is None or len(result.masks.xy) == 0:
        print(f"(!) Ảnh {img_idx}: YOLO không detect")
        return None, None

    # TÌM MASK LỚN NHẤT (dựa trên diện tích bounding box) để tránh nhận diện nhầm đốm hay bóng hình chuối thành chuối 
    boxes_np = result.boxes.xyxy.cpu().numpy()
    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes_np]
    idx = np.argmax(areas)

    # MASK YOLO: Dùng tọa độ Polygon
    polygon = result.masks.xy[idx].astype(np.int32)
    mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)

    x1, y1, x2, y2 = result.boxes.xyxy[idx].cpu().numpy().astype(int)

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_orig, x2), min(h_orig, y2)

    banana_only = cv2.bitwise_and(img, img, mask=mask)
    cropped = banana_only[y1:y2, x1:x2]

    if cropped.size == 0:
        print(f"(!) Ảnh {img_idx}: crop lỗi (kích thước = 0)")
        return None, None

    ch, cw = cropped.shape[:2]
    max_side = max(ch, cw)

    # Khởi tạo ảnh vuông nền đen (PADDING)
    square_img = np.zeros((max_side, max_side, 3), dtype=np.uint8)
    y_off, x_off = (max_side - ch) // 2, (max_side - cw) // 2
    square_img[y_off:y_off+ch, x_off:x_off+cw] = cropped

    final_img = cv2.resize(square_img, (224, 224), interpolation=cv2.INTER_AREA)

    # Crop và pad cho mask tương tự như ảnh
    mask_crop = mask[y1:y2, x1:x2]
    square_mask = np.zeros((max_side, max_side), dtype=np.uint8)
    square_mask[y_off:y_off+ch, x_off:x_off+cw] = mask_crop

    # Resize mask (KHÔNG dùng INTER_AREA, dùng NEAREST )
    final_mask = cv2.resize(square_mask, (224, 224), interpolation=cv2.INTER_NEAREST)

    # ================= DEBUG =================
    if debug:
        fig, axs = plt.subplots(1, 5, figsize=(16,4))
        axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Original")

        axs[1].imshow(mask, cmap='gray')
        axs[1].set_title("Mask")

        axs[2].imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        axs[2].set_title("Cropped")

        axs[3].imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
        axs[3].set_title("Final 224")

        overlay = final_img.copy()
        overlay[final_mask == 0] = [0, 0, 255]

        axs[4].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axs[4].set_title("Test mask after resize")

        for ax in axs:
            ax.axis("off")

        plt.suptitle(f"YOLO | img_idx={img_idx}")
        plt.show()

        print("final image size:", final_img.shape)

    return final_img, final_mask