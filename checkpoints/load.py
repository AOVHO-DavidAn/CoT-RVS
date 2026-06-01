from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch

# load model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Ricky06662/Seg-Zero-7B",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
    cache_dir="weights-cache"
)
processor = AutoProcessor.from_pretrained(
    "Ricky06662/Seg-Zero-7B", 
    padding_side="left",
    cache_dir="weights-cache"
)