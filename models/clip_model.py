import cv2
import os
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPVisionModelWithProjection
from transformers import logging
from utils.utils import new_cd
logging.set_verbosity_error()

parent_dir = os.path.dirname(__file__)

class FeatureExtractor():
    def __init__(self, args):
        self.beta = args.beta
        with new_cd(parent_dir):
            self.processor = CLIPProcessor.from_pretrained(f"../checkpoints/{args.clip_version}")
            self.model = CLIPVisionModelWithProjection.from_pretrained(f"../checkpoints/{args.clip_version}")


    def __call__(self, video_path):        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_length = frame_count / fps
        sample_rate = int(fps) * self.beta

        clip_features = []
        print("Extract the clip feature.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if cap.get(cv2.CAP_PROP_POS_FRAMES) % sample_rate == 0:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                inputs = self.processor(images=image, return_tensors="pt").pixel_values
                inputs = inputs
                
                with torch.no_grad():
                    feat = self.model(inputs)['image_embeds']
                    clip_features.append(feat.cpu().numpy())
        print("Finished.")

        clip_features = np.concatenate(clip_features, axis=0)  
        return clip_features, video_length
