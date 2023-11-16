from huggingface_hub import snapshot_download
from bigdl.llm import optimize_model
import whisper

# Clip
snapshot_download(repo_id='openai/clip-vit-base-patch32',
                  local_dir="./checkpoints/clip-vit-base-patch32")

# LLM
snapshot_download(repo_id="THUDM/chatglm3-6b-32k",
                  local_dir="./checkpoints/chatglm3-6b-32k")
snapshot_download(repo_id="THUDM/chatglm3-6b",
                  local_dir="./checkpoints/chatglm3-6b")

# Embeddings
snapshot_download(repo_id='intfloat/multilingual-e5-large',
                  local_dir="./checkpoints/multilingual-e5-large")

# Whisper
# large
model = whisper.load_model('large', download_root='./checkpoints/whisper-large')
# medium
model = whisper.load_model('medium', download_root='./checkpoints/whisper-medium')