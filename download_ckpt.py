from huggingface_hub import snapshot_download

# Clip
snapshot_download(repo_id='openai/clip-vit-base-patch32',
                  local_dir="./checkpoints/clip-vit-base-patch32")

# LLM
snapshot_download(repo_id='meta-llama/Llama-2-7b-chat-hf',
                  local_dir="./checkpoints/Llama-2-7b-chat-hf", token=hf_token)

# Translation
snapshot_download(repo_id='Helsinki-NLP/opus-mt-en-zh',
                  local_dir="./checkpoints/Helsinki-NLP-opus-mt-en-zh")
snapshot_download(repo_id='Helsinki-NLP/opus-mt-zh-en',
                  local_dir="./checkpoints/Helsinki-NLP-opus-mt-zh-en")

# Embeddings
snapshot_download(repo_id='sentence-transformers/all-MiniLM-L12-v2',
                  local_dir="./checkpoints/all-MiniLM-L12-v2")