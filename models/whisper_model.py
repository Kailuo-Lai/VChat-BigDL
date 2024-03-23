import os
import whisper
from bigdl.llm import optimize_model
from bigdl.llm.optimize import load_low_bit
from utils.utils import new_cd

parent_dir = os.path.dirname(__file__)

def has_intersection(t1, t2):
    if t1[1] < t2[0] or t2[1] < t1[0]:
        return False
    else:
        return True

class AudioTranslator():
    def __init__(self, args):
        with new_cd(parent_dir):
            model_state_file = os.listdir(f"../checkpoints/whisper-{args.whisper_version}")[0]
            self.model = whisper.load_model(f"../checkpoints/whisper-{args.whisper_version}/{model_state_file}")
            if args.whisper_low_bit:
                self.model = load_low_bit(self.model, f"../checkpoints/whisper-{args.whisper_version}-optimized")
            else:
                self.model = optimize_model(self.model)

    def __call__(self, video_path):
        print("Extract the audio results.")
        audio_results = self.model.transcribe(video_path, task = 'translate')["segments"]
        print("Finished.")
        return audio_results
    
    def match(self, audio_results, start, end):
        transcript = ''
        for res in audio_results:
            if has_intersection((start, end), (res["start"], res["end"])):
                transcript += res['text'] + ' '
        return transcript