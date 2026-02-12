from openai import AzureOpenAI
import base64
from mimetypes import guess_type
import ast
from utils.util import preprocess_prompt
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

def prompt_openai(client, model, data_url,query,num_keyframes):
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":"You are a helpful assistant that answers question in chain of thoughts."},
            {"role":"user","content":[  
                { 
                    "type": "text", 
                    "text": preprocess_prompt(query, num_keyframes)
                },
                { 
                    "type": "image_url",
                    "image_url": {
                        "url": data_url
                    }
                }
            ]
            },
        ],
        max_tokens=2500
    )
    return response.choices[0].message.content
def prompt_openai_without_cot(client, model,data_url,query,num_keyframes):
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":"You are a helpful assistant that answers question in chain of thoughts."},
            {"role":"user","content":[  
                { 
                    "type": "text", 
                    "text": f"""You will act as a keyframe selection agent for a video reasoning task. During each inference, you will be given a grid image that contains multiple keyframes sampled from a long video. The keyframes are aligned from top to down or from left to right, following their temporal order. You will also be given a complex user query that implicitly or explicitly refers to one or more target objects in the video. You need to find the best keyframes for each target object, where a segmentation model can find the target object in that frame with less effort. You have to output a list of dictionary with a format "Output list: [{{object_index: 1, keyframe: k_1, object_description: <description of the object 1 in keyframe k_1>}}, {{object_index: 2, keyframe: k_2, object_description: <description of the object 2 in keyframe k_2>}}, ...]", where each element in the list is a dictionary with three items, with object index, keyframe index, object description. k is the k-th keyframe in the grid image. object_index is a numbering integer starting with 1. object_description implies the description for that object in a particular frame, helping the model to find the object in that particular frame. For example, a valid element in an output list can be like "Output list: [{{object_index: 1, keyframe: 4, object_description: "the man at the top left corner of the image"}}]". The description should include the location of the target object in the selected frame. You have to include all objects that fulfill the requirements in the user query. Include the objects even if it is only partially visible. While choosing the keyframe for any object, you should prioritize those frames where objects are not overlapped. This will help model to better recognize the object. Keep the output list in text format. Don't use json formatting. The output list begins with the prefix "Output list: ", followed by a squared bracket with multiple curly brackets. The squared bracket should be in the same line, following the format "Output list: [...]". Don't start with a new line. Do not include anything after the output list.
Here is a grid image with {num_keyframes} keyframes. The user query is "{query}". Follow the instruction and output the index of the best keyframes.
"""
                },
                { 
                    "type": "image_url",
                    "image_url": {
                        "url": data_url
                    }
                }
            ]
            },
        ],
        max_tokens=2500
    )
    return response.choices[0].message.content
def parse_gpt_output(text):
    list_outputs = text.split("Output list: ")[-1]
    # Prepare the input string for parsing
    text_input = list_outputs.replace('object_index', '"object_index"').replace('keyframe', '"keyframe"').replace('object_description', '"object_description"')

    # Convert the string to a list of dictionaries
    output = ast.literal_eval(text_input)
    return output


