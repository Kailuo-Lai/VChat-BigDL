from bigdl.llm.transformers import AutoModel, AutoModelForCausalLM
from transformers import AutoTokenizer, LlamaTokenizer
import whisper
from bigdl.llm import optimize_model

# LLM
model = AutoModel.from_pretrained('./checkpoints/chatglm3-6b-32k',
                                    load_in_low_bit="sym_int4",
                                    trust_remote_code=True)
model.save_low_bit('./checkpoints/chatglm3-6b-32k-INT4')
tokenizer = AutoTokenizer.from_pretrained('./checkpoints/chatglm3-6b-32k',
                                            trust_remote_code=True)
tokenizer.save_pretrained('./checkpoints/chatglm3-6b-32k-INT4')

# model = AutoModel.from_pretrained('./checkpoints/chatglm3-6b',
#                                     load_in_low_bit="sym_int4",
#                                     trust_remote_code=True)
# model.save_low_bit('./checkpoints/chatglm3-6b-INT4')
# tokenizer = AutoTokenizer.from_pretrained('./checkpoints/chatglm3-6b',
#                                             trust_remote_code=True)
# tokenizer.save_pretrained('./checkpoints/chatglm3-6b-INT4')

# Whisper
# large
model = whisper.load("./checkpoints/whisper-medium")
model = optimize_model(model)
model.save_low_bit("./checkpoints/whisper-large-optimized")

# medium
model = whisper.load("./checkpoints/whisper-medium")
model = optimize_model(model)
model.save_low_bit("./checkpoints/whisper-medium-optimized")

