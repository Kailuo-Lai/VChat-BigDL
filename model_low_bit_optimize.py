from bigdl.llm.transformers import AutoModel, AutoModelForCausalLM
from transformers import AutoTokenizer, LlamaTokenizer
import whisper
from bigdl.llm import optimize_model
import os

# LLM
# ChatGLM3-6b-32k
model = AutoModel.from_pretrained('./checkpoints/chatglm3-6b-32k',
                                    load_in_low_bit="sym_int4",
                                    trust_remote_code=True)
model.save_low_bit('./checkpoints/chatglm3-6b-32k-INT4')
tokenizer = AutoTokenizer.from_pretrained('./checkpoints/chatglm3-6b-32k',
                                            trust_remote_code=True)
tokenizer.save_pretrained('./checkpoints/chatglm3-6b-32k-INT4')

# ChatGLM3-6b
model = AutoModel.from_pretrained('./checkpoints/chatglm3-6b',
                                    load_in_low_bit="sym_int4",
                                    trust_remote_code=True)
model.save_low_bit('./checkpoints/chatglm3-6b-INT4')
tokenizer = AutoTokenizer.from_pretrained('./checkpoints/chatglm3-6b',
                                            trust_remote_code=True)
tokenizer.save_pretrained('./checkpoints/chatglm3-6b-INT4')

# Whisper
# large
model_state_file = os.listdir(f"./checkpoints/whisper-large")[0]
model = whisper.load_model(f"./checkpoints/whisper-large{model_state_file}")
model = optimize_model(model)
model.save_low_bit("./checkpoints/whisper-large-optimized")

# medium
model_state_file = os.listdir(f"./checkpoints/whisper-medium")[0]
model = whisper.load_model(f"./checkpoints/whisper-medium/{model_state_file}")
model = optimize_model(model)
model.save_low_bit("./checkpoints/whisper-medium-optimized")

