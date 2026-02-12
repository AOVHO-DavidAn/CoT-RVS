import os
from PIL import Image, ImageFont, ImageDraw
from pathlib import Path
from skvideo import io

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