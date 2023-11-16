import cv2

from models.whisper_model import AudioTranslator
from models.tag2text_model import ImageCaptionerDetector
from models.clip_model import FeatureExtractor
from models.kts_model import VideoSegmentor
from models.llm_model import LlmReasoner
from utils.utils import format_time


class VChat:
    
    def __init__(self, args) -> None:
        self.args = args
    
    def init_model(self):
        print('\033[1;33m' + "Initializing models...".center(50, '-') + '\033[0m')
        print('\033[1;36m' + "Initializing CLIP model...".center(50, '-') + '\033[0m')
        self.feature_extractor = FeatureExtractor(self.args)
        self.video_segmenter = VideoSegmentor(self.args)
        print('\033[1;36m' + "Initializing Tag2Text model...".center(50, '-') + '\033[0m')
        self.image_captioner_detector = ImageCaptionerDetector(self.args)
        print('\033[1;36m' + "Initializing Whisper model...".center(50, '-') + '\033[0m')
        self.audio_translator = AudioTranslator(self.args)
        print('\033[1;36m' + "Initializing LLM...".center(50, '-') + '\033[0m')
        self.llm_reasoner = LlmReasoner(self.args)
        print('\033[1;36m' + "Initializing Translate model...".center(50, '-') + '\033[0m')
        
        print('\033[1;32m' + "Model initialization finished!".center(50, '-') + '\033[0m')
        
    def video2log(self, video_path):
        clip_features, video_length = self.feature_extractor(video_path)
        seg_windows = self.video_segmenter(clip_features, video_length)
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        audio_results = self.audio_translator(video_path)

        en_log_result = []
        for start_sec, end_sec in seg_windows:
            en_log_result_tmp = ""
            
            middle_sec = (start_sec + end_sec) // 2
            middle_frame_idx = int(middle_sec * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
            ret, frame = cap.read()
            
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_caption, image_detect = self.image_captioner_detector.image_caption_detect_from_array(frame)
                audio_transcript = self.audio_translator.match(audio_results, start_sec, end_sec)
                
                en_log_result_tmp += f"When {format_time(start_sec)} - {format_time(end_sec)}\n"
                en_log_result_tmp += f"I saw {image_caption}.\n"
                en_log_result_tmp += f"I found {image_detect}."
                
                if audio_transcript != '':
                    en_log_result_tmp += f"\nI heard someone say \"{audio_transcript}\""
            en_log_result.append(en_log_result_tmp)
                
        en_log_result = "\n\n".join(en_log_result)
        print(f"\n\033[1;34mLog: \033[0m\n{en_log_result}\n")

        self.llm_reasoner.create_qa_chain(en_log_result)
        return en_log_result
        
    def chat2video(self, user_input):
            
        print("\n\033[1;32mGnerating response...\033[0m")
        answer, generated_question, source_documents, lid = self.llm_reasoner(user_input)
        print(f"\033[1;32mQuestion: \033[0m{user_input}")
        print(f"\033[1;32mAnswer: \033[0m{answer[0][1]}")
        self.clean_history()
        
        return answer, generated_question, source_documents, lid

    def clean_history(self):
        self.llm_reasoner.clean_history()
        return