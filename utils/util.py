import os
from PIL import Image, ImageFont, ImageDraw
from pathlib import Path
from skvideo import io
import ast
import re
import numpy as np

from fsss.common.logger import Logger

def save_video(
    images,
    path,
):

    # Create the parent directory if it doesn't already exist.
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    # Save the image.
    # Image.fromarray(prep_image(image)).save(path)

    writer = io.FFmpegWriter(path, 
                                     outputdict={'-pix_fmt': 'yuv420p', '-crf': '21', 
                                                 '-vf': f'setpts=1.*PTS'})
    for frame in images:
        writer.writeFrame(frame)
    writer.close()
    

def merge_episode(args, s_img, s_mask, q_img):
    
    pass


def save_masked_image(query_img, mask, ground_truth, output_dir, img_name):
    # 处理 tensor 输入：转为 numpy RGB 格式 (H, W, C)
    if hasattr(query_img, 'cpu'):
        query_img = query_img.cpu().detach()
    if hasattr(query_img, 'numpy'):
        query_img = query_img.numpy()
    
    if hasattr(mask, 'cpu'):
        mask = mask.cpu().detach()
    if hasattr(mask, 'numpy'):
        mask = mask.numpy()
        
    if hasattr(ground_truth, 'cpu'):
        ground_truth = ground_truth.cpu().detach()
    if hasattr(ground_truth, 'numpy'):
        ground_truth = ground_truth.numpy()


    # squeeze 掉 batch 维度（如有）
    if query_img.ndim == 4 and query_img.shape[0] == 1:
        query_img = query_img.squeeze(0)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    if ground_truth.ndim == 3 and ground_truth.shape[0] == 1:
        ground_truth = ground_truth.squeeze(0)

    # CHW → HWC（PyTorch tensor 通常是 [C, H, W]）
    if query_img.ndim == 3 and query_img.shape[0] == 3:
        query_img = np.transpose(query_img, (1, 2, 0))

    # 值域 [0, 1] float → [0, 255] uint8
    if query_img.dtype in (np.float32, np.float64) and query_img.max() <= 1.0:
        query_img = (query_img * 255)
    query_img = query_img.astype(np.uint8)

    mask = mask.astype(bool)
    ground_truth = ground_truth.astype(bool)

    color = np.array([255, 0, 0], dtype=np.float32)   # RGB 红色
    alpha = 1

    # 叠加半透明红色遮罩
    overlay = query_img.astype(np.float32)
    overlay[mask] = overlay[mask] * (1 - alpha) + color * alpha
    
    overlay_label = query_img.astype(np.float32)
    overlay_label[ground_truth] = overlay_label[ground_truth] * (1 - alpha) + color * alpha
    
    #将预测mask和ground_truth的叠加结果拼接在一起
    combined_overlay = np.concatenate((overlay, overlay_label), axis=1)

    # 用 PIL 保存可视化结果，并添加顶部文字标识
    vis_save_path = os.path.join(output_dir, f"{img_name}_vis.jpg")
    combined_img = Image.fromarray(combined_overlay.astype(np.uint8))

    # 添加顶部文字标识栏
    h, w = combined_overlay.shape[0], combined_overlay.shape[1]
    label_height = max(30, h // 15)  # 文字栏高度
    label_bg = Image.new('RGB', (w, label_height), (30, 30, 30))  # 深灰色背景栏

    # 创建最终画布：文字栏在上，拼接图在下
    final_img = Image.new('RGB', (w, h + label_height))
    final_img.paste(label_bg, (0, 0))
    final_img.paste(combined_img, (0, label_height))

    draw = ImageDraw.Draw(final_img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", label_height - 6)
    except (IOError, OSError):
        font = ImageFont.load_default()

    half_w = w // 2
    # 左半部分标注 "Prediction"，右半部分标注 "Ground Truth"
    draw.text((half_w // 2, 3), "Prediction", fill=(255, 255, 255), font=font,
              anchor="mt")
    draw.text((half_w + half_w // 2, 3), "Ground Truth", fill=(255, 255, 255), font=font,
              anchor="mt")

    final_img.save(vis_save_path)


def merge_keyframe(args,video_name,images):

    width, height = images[0].size
    
    index_spacing = min(width,height)//8
    spacing = min(width,height)//30
    boundary = 10
    if width > height:
    # Align vertically
        total_height = height * len(images) + spacing * (len(images) - 1) + boundary*len(images)  
        final_image = Image.new('RGB', (width + boundary + index_spacing, total_height), (255, 0, 0))
        draw = ImageDraw.Draw(final_image)
        font = ImageFont.truetype("NotoSansMath-Regular.ttf", height//12)
        
        y_offset = boundary//2  # Start with a boundary
        for idx,img in enumerate(images):
            final_image.paste(Image.new('RGB', (index_spacing, height+boundary), (255, 255, 255)), (0, y_offset-boundary//2))
            draw.text((0, y_offset+height//2-height//24),str(idx+1),(0,0,0),font=font)
            final_image.paste(img, (boundary//2+index_spacing, y_offset))
            y_offset += height + boundary//2  # Move down for the next image, adding space
            final_image.paste(Image.new('RGB', (width+boundary+index_spacing, spacing), (255, 255, 255)), (0, y_offset))  # Add space
            y_offset += spacing + boundary//2  # Move down for the next image boundary
    else:
        # Align horizontally
        total_width = width * len(images) + spacing * (len(images) - 1) + boundary*len(images)  
        final_image = Image.new('RGB', (total_width, height + boundary + index_spacing), (255, 0, 0))
        draw = ImageDraw.Draw(final_image)
        font = ImageFont.truetype("NotoSansMath-Regular.ttf", width//12)
        
        x_offset = boundary//2  # Start with a boundary
        for idx,img in enumerate(images):
            final_image.paste(Image.new('RGB', (width+boundary, index_spacing), (255, 255, 255)), (x_offset-boundary//2, 0))
            draw.text((x_offset+width//2-width//24,0),str(idx+1),(0,0,0),font=font)
            final_image.paste(img, (x_offset, boundary//2+index_spacing))
            x_offset += width + boundary//2  # Move right for the next image, adding space
            final_image.paste(Image.new('RGB', (spacing, height+boundary+index_spacing), (255, 255, 255)), (x_offset, 0))  # Add space
            x_offset += spacing + boundary//2  # Move right for the next image boundary
        
    result_image=final_image
    max_size = 3200
    if result_image.width > result_image.height:
        if result_image.width > max_size:
            scale_factor = max_size / result_image.width
            new_size = (max_size, int(result_image.height * scale_factor))
            result_image = result_image.resize(new_size, Image.Resampling.LANCZOS)
    else:
        if result_image.height > max_size:
            scale_factor = max_size / result_image.height
            new_size = (int(result_image.width * scale_factor), max_size)
            result_image = result_image.resize(new_size, Image.Resampling.LANCZOS)
    save_path = os.path.join(args.output_dir,"keyframes",f"{video_name}.jpg")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    result_image.save(save_path)
    return save_path

def preprocess_prompt(query,num_keyframes):
    
    return f"""You will act as a keyframe selection agent for a video reasoning task. During each inference, you will be given a grid image that contains multiple keyframes sampled from a long video. The keyframes are aligned from top to down or from left to right, following their temporal order. You will also be given a complex user query that implicitly or explicitly refers to one or more target objects in the video. You need to think in chain of thoughts to analyze each keyframes and find the best keyframes for each target object, where a segmentation model can find the target object in that frame with less effort. Your chain of thoughts should begin with what can be seen in each keyframe, how many objects in total fulfilling the requirements of the user query, etc. Some of the objects may be seriously obscured or blocked by other objects. Some of the objects may be camouflaged in their surroundings. Analyze each frame separately to get all the visible objects. This chain of thoughts should follow the output format: 
"Chain of Thoughts: 
- Frame 1: <analysis of frame 1>;
- Frame 2: <analysis of frame 2>;
...". For the analysis of each frame, you also have to follow the chain-of-thought format: 
"- *<question 1>* <answer 1>;
- *<question 2>* <answer2>; 
...", where you have to ask question to yourself and answer it. Your answer should be as detailed as possible. You should start with broader questions, like "what can be seen in the frame?" to some detailed questions like "are there any other objects that haven't be listed?" and "how many and which objects satisfy the user query?". There will be many questions and answers in the analysis of each frame, helping you to fully understand the frame. The actual questions varies by cases. Generate the in-depth questions and answers based on your previous analysis. Your thinking process aims to find the keyframe for each target object of interest (find all the target objects, each of which corresponds to a keyframe) related to the user query. Lastly, you have to summarize the output by outputing a list of dictionary with a format "Output list: [{{object_index: 1, keyframe: k_1, object_description: <description of the object 1 in keyframe k_1>}}, {{object_index: 2, keyframe: k_2, object_description: <description of the object 2 in keyframe k_2>}}, ...]", where each element in the list is a dictionary with three items, with object index, keyframe index, object description. k is the k-th keyframe in the grid image. object_index is a numbering integer starting with 1. object_description implies the description for that object in a particular frame, helping the model to find the object in that particular frame. It should include the location of the target object in the selected frame. For example, a valid element in an output list can be like "Output list: [{{object_index: 1, keyframe: 4, object_description: "the man at the top left corner of the image"}}]". You have to include all objects that fulfill the requirements in the user query. Include the objects even if it is only partially visible. While choosing the keyframe for any object, you should prioritize those frames where objects are not overlapped. This will help model to better recognize the object. Keep the output list in text format. Don't use json formatting. The output list begins with the prefix "Output list: ", followed by a squared bracket with multiple curly brackets. The squared bracket should be in the same line, following the format "Output list: [...]". Don't start with a new line. \n

Here is a grid image with {num_keyframes} keyframes. The user query is "{query}". Follow the instruction and output the index of the best keyframes.
"""


def preprocess_prompt_fsss(k_shot=1):
    
#     return f"""You will act as a visual reasoning agent for a few-shot semantic segmentation task. During each inference, you will be given a single concatenated image representing a {k_shot}-shot task. The image panels are aligned from left to right. For example, in a 1-shot setting, it consists of: Support Image, Target Highlight (the target object overlaid with a semi-transparent mask), Binary Mask, and finally the Query Image. There is no explicit text query; your target category is implicitly defined by the object highlighted in the Support panels.

# You need to think in chain of thoughts to first deduce the semantic category and core visual characteristics of the target object from the Support set, and then find all instances of this exact semantic category in the Query Image. Your chain of thoughts should begin with analyzing the support target (what it is, shape, texture, material, parts), followed by a detailed scan of the Query Image. Some instances in the Query Image may be seriously obscured, blocked by other objects, camouflaged, or appear in different scales, angles, and lighting conditions compared to the support object.

# This chain of thoughts should follow the output format: 
# "Chain of Thoughts: 
# - Support Set Analysis: <analysis of the support target>;
# - Query Image Analysis: <analysis of the query image>;"

# For the analysis of each section, you also have to follow the chain-of-thought format: 
# "- *<question 1>* <answer 1>;
# - *<question 2>* <answer 2>; 
# ...", where you have to ask questions to yourself and answer them. Your answers should be as detailed as possible. You should start with broader questions, like "What specific object or part is highlighted in the support image?" and "What are its key visual features?" to some detailed questions in the query analysis like "Which objects in the query image share these semantic features?" and "Are there any partial or occluded instances that haven't been listed?". The actual questions vary by cases. Generate in-depth questions and answers based on your previous analysis. Your thinking process aims to find every valid instance of the target category in the Query Image.

# Lastly, you have to summarize the output by outputting a list of dictionary with a format "Output list: [{{instance_index: 1, instance_description: <detailed description of instance 1>}}, {{instance_index: 2, instance_description: <detailed description of instance 2>}}, ...]", where each element in the list is a dictionary with two items: instance_index and instance_description. instance_index is a numbering integer starting with 1. instance_description is a detailed text description of that specific instance in the Query Image, helping a downstream referring segmentation model to accurately find and segment it. The description MUST include its exact spatial location in the Query Image (e.g., top-left, bottom-center) and contextual visual details (e.g., 'the partially rusted metal ring located in the upper right corner, partially obscured by a wire'). You have to include all instances that belong to the support category. Include the objects even if they are only partially visible. Keep the output list in text format. Don't use json formatting. The output list begins with the prefix "Output list: ", followed by a squared bracket with multiple curly brackets. The squared bracket should be in the same line, following the format "Output list: [...]". Don't start with a new line. Do not include anything after the output list.

# Here is a concatenated image for a {k_shot}-shot task. Follow the instruction and output the list of target instances found in the Query Image.
# """


    return f"""You will act as a visual reasoning agent for a few-shot semantic segmentation task. During each inference, you will be given a single concatenated image representing a {k_shot}-shot task. The image panels are aligned from left to right. For example, in a 1-shot setting, it consists of: Support Image, Target Object Cutout (the target object isolated on a white background with its original surroundings removed), Binary Mask, and finally the Query Image. There is no explicit text query; your target category is implicitly defined by the object isolated in the Support panels.

You need to think in chain of thoughts to first deduce the BROAD SEMANTIC CATEGORY (e.g., "chair", "car", "dog") of the target object from the Support set, and then find all instances belonging to this general semantic category in the Query Image. 

Your task is category-level semantic segmentation, NOT strict instance matching. Do NOT restrict your search to objects with the exact same color, material, style, or sub-type as the support object. You must generalize to the broader class. 
For example, if the support image highlights a "red wooden dining chair", you MUST find ALL types of chairs in the query image (e.g., blue plastic office chairs, metal folding chairs, wooden stools), because they all belong to the overarching semantic category of "chair".

Note that although the target instances in the Query Image belong to the exact same broad semantic category as the Support target, they may exhibit extreme differences in fine-grained visual details. They might appear in entirely different colors, textures, materials, sub-types, scales, viewing angles, physical poses, or lighting conditions compared to the support object.

You need to think in chain of thoughts to first deduce the semantic category and core visual characteristics of the target object from the Support set, and then find all instances of this exact semantic category in the Query Image. 

Note that although the target instances in the Query Image belong to the exact same semantic category as the Support target, they may exhibit significant differences in fine-grained visual details. They might appear in entirely different scales, viewing angles, physical poses, or lighting conditions compared to the support object.

Your chain of thoughts should begin with analyzing the support target (what it is, shape, texture, parts), followed by a detailed scan of the Query Image. Some instances in the Query Image may be seriously obscured, blocked by other objects, or camouflaged in complex backgrounds.

This chain of thoughts should follow the output format: 
"Chain of Thoughts: 
- Support Set Analysis: <analysis of the support target>;
- Query Image Analysis: <analysis of the query image>;"

For the analysis of each section, you also have to follow the chain-of-thought format: 
"- *<question 1>* <answer 1>;
- *<question 2>* <answer 2>; 
...", where you have to ask questions to yourself and answer them. Your answers should be as detailed as possible. You should start with broader questions, like "What specific object or part is highlighted in the support image?" and "What are its key visual features?" to some detailed questions in the query analysis like "Which objects in the query image share these semantic features?" and "Are there any partial or occluded instances that haven't been listed?". The actual questions vary by cases. Generate in-depth questions and answers based on your previous analysis. Your thinking process aims to find every valid instance of the target category in the Query Image.

CRITICAL INSTRUCTION FOR OCCLUDED OBJECTS: 
If a single target instance in the Query Image is severely occluded such that it is visually separated into multiple disconnected visible fragments, you MUST treat each disconnected visible fragment as a SEPARATE entity in your output list. Do not group them into one description. 

Crucially, you must ask: "How can I uniquely distinguish each found instance from the others?" 
The actual questions vary by cases. Generate in-depth questions and answers based on your previous analysis. Your thinking process aims to find every valid instance of the target category in the Query Image.

CRITICAL INSTRUCTION FOR MULTIPLE INSTANCES: 
When there are multiple instances (or multiple disconnected fragments) of the target category in the Query Image, your descriptions MUST be highly discriminative and mutually exclusive. Do not use generic descriptions that could apply to more than one instance. To prevent the downstream segmentation model from confusing them, you must explicitly use unique spatial anchors (e.g., 'the one closest to the red car', 'in the absolute top-right corner') and unique distinguishing attributes (e.g., 'the only one looking left', 'darker than the other instance').

Lastly, you have to summarize the output by outputting a list of dictionary with a format "Output list: [{{instance_index: 1, instance_description: <detailed description of instance 1>}}, {{instance_index: 2, instance_description: <detailed description of instance 2>}}, ...]", where each element in the list is a dictionary with two items: instance_index and instance_description. instance_index is a numbering integer starting with 1. instance_description is a detailed text description of that specific instance (or visible fragment) in the Query Image, helping a downstream referring segmentation model to accurately find and segment it. The description MUST include its exact spatial location in the Query Image (e.g., top-left, bottom-center) and contextual visual details. 

You have to include all instances and fragments that belong to the support category. Keep the output list in text format. Don't use json formatting. The output list begins with the prefix "Output list: ", followed by a squared bracket with multiple curly brackets. The squared bracket should be in the same line, following the format "Output list: [...]". Don't start with a new line. Do not include anything after the output list.

For example, if the target is a dog, and a dog in the Query Image is standing behind a solid wooden fence causing its visible body to be visually cut into two disconnected parts (the head and the tail), you should output two separate elements like:
{{instance_index: 1, instance_description: "the visible head and front paws of the dog, located in the lower-left of the image, emerging from the left side of the fence"}} and 
{{instance_index: 2, instance_description: "the visible tail and hind legs of the same dog, located in the lower-right of the image, sticking out from the right side of the fence"}}.

Here is a concatenated image for a {k_shot}-shot task. Follow the instruction and output the list of target instances found in the Query Image.
"""
    
    
#     return f"""You will act as a visual reasoning agent for a few-shot semantic segmentation task. During each inference, you will be given a single concatenated image representing a {k_shot}-shot task. The image panels are aligned from left to right. For example, in a 1-shot setting, it consists of: Support Image, Target Highlight (the target object overlaid with a semi-transparent mask), Binary Mask, and finally the Query Image. There is no explicit text query; your target category is implicitly defined by the object highlighted in the Support panels.

# You need to think in chain of thoughts to first deduce the semantic category and core visual characteristics of the target object from the Support set, and then find all instances of this exact semantic category in the Query Image. 

# Note that although the target instances in the Query Image belong to the exact same semantic category as the Support target, they may exhibit significant differences in fine-grained visual details. They might appear in entirely different scales, viewing angles, physical poses, or lighting conditions compared to the support object.

# Your chain of thoughts should begin with analyzing the support target (what it is, shape, texture, parts), followed by a detailed scan of the Query Image. Some instances in the Query Image may be seriously obscured, blocked by other objects, or camouflaged in complex backgrounds.

# This chain of thoughts should follow the output format: 
# "Chain of Thoughts: 
# - Support Set Analysis: <analysis of the support target>;
# - Query Image Analysis: <analysis of the query image>;"

# For the analysis of each section, you also have to follow the chain-of-thought format: 
# "- *<question 1>* <answer 1>;
# - *<question 2>* <answer 2>; 
# ...", where you have to ask questions to yourself and answer them. Your answers should be as detailed as possible. You should start with broader questions, like "What specific object or part is highlighted in the support image?" and "What are its key visual features?" to some detailed questions in the query analysis like "Which objects in the query image share these semantic features?" and "Are there any partial or occluded instances that haven't been listed?". 

# [新增：强制在思维链中进行对比分析]
# Crucially, you must ask: "How can I uniquely distinguish each found instance from the others?" 

# The actual questions vary by cases. Generate in-depth questions and answers based on your previous analysis. Your thinking process aims to find every valid instance of the target category in the Query Image.

# [新增：多实例排他性描述指令]
# CRITICAL INSTRUCTION FOR MULTIPLE INSTANCES: 
# When there are multiple instances (or multiple disconnected fragments) of the target category in the Query Image, your descriptions MUST be highly discriminative and mutually exclusive. Do not use generic descriptions that could apply to more than one instance. To prevent the downstream segmentation model from confusing them, you must explicitly use unique spatial anchors (e.g., 'the one closest to the red car', 'in the absolute top-right corner') and unique distinguishing attributes (e.g., 'the only one looking left', 'darker than the other instance'). 

# CRITICAL INSTRUCTION FOR OCCLUDED OBJECTS: 
# If a single target instance in the Query Image is severely occluded such that it is visually separated into multiple disconnected visible fragments, you MUST treat each disconnected visible fragment as a SEPARATE entity in your output list. Do not group them into one description. 
# For example, if the target is a dog, and a dog in the Query Image is visually cut into two disconnected parts (the head and the tail) by a fence, you should output two separate elements like:
# {{instance_index: 1, instance_description: "the visible head of the dog, located in the lower-left, emerging from the left side of the fence. This part only contains the head and ears, no body."}} and 
# {{instance_index: 2, instance_description: "the visible tail of the same dog, located in the lower-right, sticking out from the right side of the fence. This part only contains a furry tail, clearly separated from the head section."}}.

# Lastly, you have to summarize the output by outputting a list of dictionary with a format "Output list: [{{instance_index: 1, instance_description: <detailed description of instance 1>}}, {{instance_index: 2, instance_description: <detailed description of instance 2>}}, ...]", where each element in the list is a dictionary with two items: instance_index and instance_description. instance_index is a numbering integer starting with 1. instance_description is a detailed text description of that specific instance (or visible fragment) in the Query Image, helping a downstream referring segmentation model to accurately find and segment it. The description MUST include its exact spatial location in the Query Image (e.g., top-left, bottom-center) and contextual visual details. 

# You have to include all instances and fragments that belong to the support category. Keep the output list in text format. Don't use json formatting. The output list begins with the prefix "Output list: ", followed by a squared bracket with multiple curly brackets. The squared bracket should be in the same line, following the format "Output list: [...]". Don't start with a new line. Do not include anything after the output list.

# Here is a concatenated image for a {k_shot}-shot task. Follow the instruction and output the list of target instances found in the Query Image.
# """



def parse_gpt_output(text):
    # 1. 提取 "Output list: " 之后的所有内容
    list_outputs = text.split("Output list: ")[-1]

    # 2. 找到第一个 [...] 块（列表），用括号深度匹配，忽略后面的自然语言
    start = list_outputs.find('[')
    if start == -1:
        return []

    depth = 0
    end = -1
    for i in range(start, len(list_outputs)):
        if list_outputs[i] == '[':
            depth += 1
        elif list_outputs[i] == ']':
            depth -= 1
            if depth == 0:
                end = i
                break

    if end == -1:
        return []

    text_input = list_outputs[start:end + 1]

    # 3. 规范化未加引号的 key
    keys = [
        "object_index",
        "keyframe",
        "object_description",
        "instance_index",
        "instance_description",
    ]
    for key in keys:
        text_input = re.sub(rf'(?<!\")\b{key}\b(?!\")', f'"{key}"', text_input)

    # 4. 解析为 Python 字面量
    try:
        output = ast.literal_eval(text_input)
    except (SyntaxError, ValueError):
        Logger.info("WARNING: Failed to parse GPT output. Returning empty list.")
        return []

    # 5. 将 instance_* 归一化为 object_*
    normalized = []
    for item in output:
        if isinstance(item, dict):
            if "object_index" not in item and "instance_index" in item:
                item["object_index"] = item.pop("instance_index")
            if "object_description" not in item and "instance_description" in item:
                item["object_description"] = item.pop("instance_description")
        normalized.append(item)
    return normalized