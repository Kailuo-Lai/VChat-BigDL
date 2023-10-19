# 🤖 VChat-BigDL: A ChatBot for Video Understanding Based on BigDL

Powered by BigDL, Llama, Clip, Whisper, Tag2Text, Helsinki, LangChain and inspired by showlab/log, we turn a video into a long document which records visual and audio information. Then we can chat over the record only using Intel CPU.

## Environment Preparing

### Download Whisper

```bash
pip install git+https://github.com/openai/whisper.git
```
### Download Tag2Text weight

<https://huggingface.co/spaces/xinyu1205/recognize-anything/blob/main/tag2text_swin_14m.pth>

## Acknowledge

This project is based on [BigDL](https://github.com/intel-analytics/BigDL), [Vlog](https://github.com/showlab/VLog/tree/main), [Tag2Text](https://tag2text.github.io/), [Whisper](https://github.com/openai/whisper), [Llama2](https://github.com/facebookresearch/llama), [Helsinki](https://huggingface.co/Helsinki-NLP), [KTS](https://inria.hal.science/hal-01022967/PDF/video_summarization.pdf), [LangChain](https://python.langchain.com/en/latest/).