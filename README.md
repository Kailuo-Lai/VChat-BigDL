# 🤖 VChat-BigDL: A ChatBot for Video Understanding Based on BigDL

Powered by BigDL, Llama, Clip, Whisper, Tag2Text, Helsinki, LangChain and inspired by showlab/log, we turn a video into a long document which records visual and audio information. Then we can chat over the record only using Intel CPU.

## Environment Preparing

### System ---Windows

#### 1. Create Conda Environment

```bash
conda  create -n vchat python=3.9 -y
activate vchat
pip install -r ./requirements.txt
```
#### 2. Install ffmpeg

```bash
conda install -c conda-forge ffmpeg -y
```

#### 3. Download model weight

##### Download Clip, Llama, Helsinki, all-MiniLM-L12-v2 by huggingface

```bash
python download_ckpt.py
```

##### Download Tag2Text weight

<https://huggingface.co/spaces/xinyu1205/recognize-anything/blob/main/tag2text_swin_14m.pth>


**✅Please save weights to ./checkpoints**

#### 4. Optimaize llm

```bash
python LLM_low_bit_optimize.py
```

#### 5. run with gradio
```bash
python main_gradio.py
```

#### ❗Attention❗

if you want to share the link
```bash
Could not create share link. Missing file: D:\anaconda3\envs\vchat\lib\site-packages\gradio\frpc_windows_amd64_v0.2.
```

Please check your internet connection. This can happen if your antivirus software blocks the download of this file. You can install manually by following these steps:

1. Download this file: https://cdn-media.huggingface.co/frpc-gradio-0.2/frpc_windows_amd64.exe
2. Rename the downloaded file to: frpc_windows_amd64_v0.2
3. Move the file to this location: .\anaconda3\envs\vchat\lib\site-packages\gradio

## Acknowledge

This project is based on [BigDL](https://github.com/intel-analytics/BigDL), [Vlog](https://github.com/showlab/VLog/tree/main), [Tag2Text](https://tag2text.github.io/), [Whisper](https://github.com/openai/whisper), [Llama2](https://github.com/facebookresearch/llama), [Helsinki](https://huggingface.co/Helsinki-NLP), [KTS](https://inria.hal.science/hal-01022967/PDF/video_summarization.pdf), [LangChain](https://python.langchain.com/en/latest/).