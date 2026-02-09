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