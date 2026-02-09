
from LLaVA.llava.eval.run_llava import eval_model

def parse_gpt_output(answer):
    #print(answer.split("This is because")[0][-5:])
    return True if 'Yes.' in answer.split("This is because")[0][-5:] else False

def ask_llava(model, image_path, query, **kwargs):
    prompt = f"Consider the query \"{query}\" for an object tracking task. First analyze what can be seen in the input image and then output a simple answer with Yes or No to justify whether the input image is suitable as a keyframe. A good keyframe is an image where the target object can be seen. Please follow the output format \"The image contains .... Therefore, the justification of using this image as keyframe is <Yes./No.>\"."
    
    llava_args = type('Args', (), {
        "model_base": None,
        "model_path": kwargs["model_path"],
        "pretrained_model": model,
        "query": prompt,
        "conv_mode": None,
        "image_file": image_path,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()
    answer = eval_model(llava_args)
        
    return answer
        