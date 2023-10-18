import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from utils.utils import new_cd

parent_dir = os.path.dirname(__file__)

class Translator:
    
    def __init__(self, convert_lid) -> None:
        """
        convert_lid: 'en-zh', 'zh-en'...
        """
        with new_cd(parent_dir):
            self.tokenizer = AutoTokenizer.from_pretrained(f"../checkpoints/Helsinki-NLP-opus-mt-{convert_lid}")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(f"../checkpoints/Helsinki-NLP-opus-mt-{convert_lid}")

    def __call__(self, text):
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids=input_ids)
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return outputs
        