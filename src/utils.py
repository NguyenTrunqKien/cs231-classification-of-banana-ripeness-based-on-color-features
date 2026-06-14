import os
import cv2
import matplotlib.pyplot as plt

# Hàm đọc dữ liệu từ thư mục folder
def load_images_from_folder(folder):
    images = []
    labels = []
    class_names = sorted(os.listdir(folder))

    for label, class_name in enumerate(class_names):
        class_folder = os.path.join(folder, class_name)
        if os.path.isdir(class_folder):
            for filename in sorted(os.listdir(class_folder)):
                img_path = os.path.join(class_folder, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)
                    labels.append(label)

    return images, labels, class_names

# Hàm hiển thị một ảnh
def show_image(image, title="Image", size=5):
    """
    Hàm hiển thị một ảnh duy nhất.
    - image: Mảng ảnh (numpy array).
    - title: Tiêu đề ảnh.
    - size: Kích thước hiển thị (inch).
    """
    plt.figure(figsize=(size, size))

    # Chuyển đổi màu nếu ảnh có 3 kênh (thường là BGR từ OpenCV)
    if len(image.shape) == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
    else:
        # Hiển thị ảnh xám (grayscale)
        plt.imshow(image, cmap='gray')

    plt.title(title)
    plt.axis('off')
    plt.show()
# Cách dùng:
# show_image(train_images[0], title="Ảnh đầu tiên", size=6)

# Hàm hiển thị các ảnh tăng cường
def show_augmentations(augmented_pairs, title="Data Augmentation Results"):
    """
    Hiển thị danh sách các cặp ảnh và mask đã được tăng cường.
    """
    n = len(augmented_pairs)
    fig, axes = plt.subplots(2, n, figsize=(n * 3, 6))

    for i in range(n):
        img, mask = augmented_pairs[i]

        # Hiển thị ảnh (Chuyển BGR sang RGB để matplotlib hiển thị đúng màu)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[0, i].imshow(img_rgb)
        if i == 0:
          axes[0, i].set_title(f"Original")
        elif i == 2:
          axes[0, i].set_title(f"Flip Horizontal")
        elif i == 1:
          axes[0, i].set_title(f"Noise")
        elif i == 3:
          axes[0, i].set_title(f"Brighten")
        elif i == 4:
          axes[0, i].set_title(f"Darken")

        axes[0, i].axis('off')

        # Hiển thị mask
        axes[1, i].imshow(mask, cmap='gray')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1.05)
    plt.show()

# --- VÍ DỤ SỬ DỤNG ---
# Giả sử bạn đã có final_img và final_mask từ hàm process_banana_224
# result_pairs = augment_image(final_img, final_mask)
# show_augmentations(result_pairs)

