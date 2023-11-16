from huggingface_hub import snapshot_download
from bigdl.llm import optimize_model
import whisper

# Clip
snapshot_download(repo_id='openai/clip-vit-base-patch32',
                  local_dir="./checkpoints/clip-vit-base-patch32")

# LLM
snapshot_download(repo_id="THUDM/chatglm3-6b-32k",
                  local_dir="./checkpoints/chatglm3-6b-32k")
# snapshot_download(repo_id="THUDM/chatglm3-6b",
#                   local_dir="./checkpoints/chatglm3-6b")

# Translation
snapshot_download(repo_id='Helsinki-NLP/opus-mt-en-zh',
                  local_dir="./checkpoints/Helsinki-NLP-opus-mt-en-zh")

# Embeddings
snapshot_download(repo_id='sentence-transformers/all-MiniLM-L12-v2',
                  local_dir="./checkpoints/all-MiniLM-L12-v2")
snapshot_download(repo_id='BAAI/bge-small-zh-v1.5',
                  local_dir="./checkpoints/bge-small-zh-v1.5")

# Whisper

# large
model = whisper.load_model('large', download_root='./checkpoints/whisper-large')
# medium
model = whisper.load_model('medium', download_root='./checkpoints/whisper-medium')