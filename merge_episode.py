import cv2
import numpy as np
import os

def create_mllm_test_image(support_path, mask_path, query_path, output_dir="./outputs/merged_episode", img_name = "mllm_prompt_input.jpg"):
    # 1. 读取图像
    support_img = cv2.imread(support_path)
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # Mask以单通道灰度图读取
    query_img = cv2.imread(query_path)

    # 检查图像是否成功读取
    if support_img is None or mask_img is None or query_img is None:
        raise ValueError("请检查图片路径是否正确，部分图片未读取成功。")

    # 2. 统一图像高度 (按比例缩放，防止物体变形)
    # 设定一个固定的高度，例如 512 像素
    target_height = 512
    
    def resize_to_height(img, target_h):
        h, w = img.shape[:2]
        new_w = int(w * (target_h / h))
        return cv2.resize(img, (new_w, target_h))

    support_img = resize_to_height(support_img, target_height)
    mask_img = resize_to_height(mask_img, target_height)
    query_img = resize_to_height(query_img, target_height)

    # 3. 将单通道 Mask 转换为三通道，以便与其他彩色图像拼接
    mask_3c = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)

    # 4. 生成 Support Image 的半透明高亮叠加图 (Overlay)
    # 将 Mask 进行二值化处理，确保只有 0 和 255
    _, binary_mask = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
    # 转换为布尔型掩码
    bool_mask = binary_mask > 0
    
    # 复制原图用于叠加
    overlay_img = support_img.copy()
    # 红色高亮 (BGR格式: [0, 0, 255])
    overlay_color = np.array([0, 0, 255]) 
    
    # 仅在目标区域进行颜色混合，比例为 0.5 原图 + 0.5 红色
    overlay_img[bool_mask] = overlay_img[bool_mask] * 0.5 + overlay_color * 0.5
    overlay_img = overlay_img.astype(np.uint8)

    # 5. 为每张图片添加顶部文字标签区块
    def add_text_header(img, text):
        h, w = img.shape[:2]
        # 创建一个高度为 60 的白色区块作为头部
        header = np.ones((60, w, 3), dtype=np.uint8) * 255
        # 在白色区块上写入黑色文字
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(header, text, (10, 40), font, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
        # 垂直拼接头部和图片 (上下拼接)
        return cv2.vconcat([header, img])

    support_img_with_text = add_text_header(support_img, "1. Support Image")
    overlay_img_with_text = add_text_header(overlay_img, "2. Target Highlight")
    mask_3c_with_text = add_text_header(mask_3c, "3. Binary Mask")
    query_img_with_text = add_text_header(query_img, "4. Query Image")

    # 6. 水平拼接所有处理好的图片 (左右拼接)
    # cv2.hconcat 要求所有图像高度一致，我们在前面已经保证了这一点
    final_mllm_image = cv2.hconcat([
        support_img_with_text, 
        overlay_img_with_text, 
        mask_3c_with_text, 
        query_img_with_text
    ])

    # 7. 保存结果
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, img_name)
    cv2.imwrite(output_path, final_mllm_image)
    print(f"拼接图像已成功保存至: {output_path}")
    return final_mllm_image

if __name__ == "__main__":
    # Example usage for quick manual checks.
    episode_dir = "./few-shot_episode/balloon"
    support_path = os.path.join(episode_dir, "support_img.jpg")
    mask_path = os.path.join(episode_dir, "support_mask.png")
    query_path = os.path.join(episode_dir, "query_img.jpg")

    create_mllm_test_image(support_path, mask_path, query_path, "./outputs/merged_episode", "test_input.jpg")