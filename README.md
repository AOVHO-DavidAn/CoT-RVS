# CoT-RVS: Zero-Shot Chain-of-Thought Reasoning Segmentation for Videos [ICLR 2026]
> [**CoT-RVS: Zero-Shot Chain-of-Thought Reasoning Segmentation for Videos**](https://arxiv.org/abs/2505.18561)      
> Shiu-hong Kao, Yu-Wing Tai, Chi-Keung Tang       
> [Project page](https://danielshkao.github.io/cot-rvs.html)

---
<img width="649" height="385" alt="image" src="https://github.com/user-attachments/assets/b462753c-0f75-4a87-9f2b-965696ab134d" />

## News

- [01/2026] CoT-RVS is accepted to ICLR 2026! &#127881;&#127881;
- [09/2025] Preprint is abailable on [arXiv](https://arxiv.org/abs/2505.18561).
  
## To-do List

Our code will be released soon. Please stay tuned!&#129395;
- [x] CoT-RVS-GPT-4o
- [ ] CoT-RVS-Gemma
- [x] Chat with CoT-RVS-LLaVA (online)
- [ ] T-ReasonVOS dataset

## Installation

Our code is tested on [Ubuntu 22.04](https://releases.ubuntu.com/jammy/) with with Python 3.10 and CUDA 12.2.
```
git clone https://github.com/DanielSHKao/CoT-RVS.git
cd CoT-RVS
conda create -n cot-rvs -y python=3.10
conda activate cot-rvs

# (Optional) Install LLaVA only for online extension of CoT-RVS
git clone https://github.com/haotian-liu/LLaVA && cd LLaVA
pip install -e .
cd ..

# This is required
pip install -r requirements.txt

# Install flash-attn. Our code is tested on flash_attn==2.7.3.
pip install flash-attn --no-build-isolation
# Install directly from pre-compiled wheel (we use this during the testing):
# pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

## Download Models

1. Download [SAM2 checkpoints](https://github.com/facebookresearch/sam2?tab=readme-ov-file#download-checkpoints) as video processor. We use [sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt) in our paper.
2. Download Seg-Zero checkpoints from [Huggingface](https://huggingface.co/Ricky06662/Seg-Zero-7B).
3. (Optional) Download LLaVA1.5-7B from [Huggingface](https://huggingface.co/liuhaotian/llava-v1.5-7b) for Online Reasoning VOS.

## Inference

### Chat with CoT-RVS
To chat with CoT-RVS, please first fill in your OpenAI API Key in `config/openai.yaml`. Then, run the following command:
```
python chat_offline \
--sam2_model [path to SAM2 checkpoint] \
--segzero_model [path to SegZero checkpoint] \
--output_dir [output directory] \
--num_candidates [number of keyframe candidates]
```

### Chat with CoT-RVS-(Online extension)
To chat with CoT-RVS-LLaVA for online Reasoning VOS, we use the `eval_model` method in LLaVA's implementation. 
1. Please first modify the `eval_model` in `LLaVA\llava\eval\run_llava.py`:
```
> ...
> outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
> # print(outputs) # Comment this line
> return outputs # Add this line to return LLaVA's response
```
2. Run the following command:
```
python chat_online \
--sam2_model [path to SAM2 checkpoint] \
--segzero_model [path to SegZero checkpoint] \
--llava_model [path to LLaVA checkpoint] \
--output_dir [output directory] \
--xi 10
```
You may adjust the hyper-parameter `--xi` to change LLaVA's intervention frequency.

## T-ReasonVOS Dataset
We will release the T-ReasonVOS dataset very soon. Please stay tuned!

## Citation
If you find this repository helpful, please consider citing:
```
@article{CoTRVS,
  title={CoT-RVS: Zero-Shot Chain-of-Thought Reasoning Segmentation for Videos},
  author={Kao, Shiu-hong and Tai, Yu-Wing and Tang, Chi-Keung},
  journal={arXiv preprint arXiv:2505.18561},
  year={2025}
}
```
