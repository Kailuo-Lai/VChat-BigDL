import torch
import torchvision.transforms as transforms
from bigdl.llm import optimize_model

from PIL import Image
from models.tag2text_src.tag2text import tag2text_caption
from utils.utils import new_cd

class ImageCaptionerDetector:
    def __init__(self, args):
        self.threshold = args.tag2text_thershld # threshold for tagging, default 0.68
        self.init_model()
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        ])
    
    def init_model(self):
        # delete some tags that may disturb captioning
        # 127: "quarter"; 2961: "back", 3351: "two"; 3265: "three"; 3338: "four"; 3355: "five"; 3359: "one"
        delete_tag_index = [127,2961, 3351, 3265, 3338, 3355, 3359]

        #######load model
        with new_cd('./models'):
            model = tag2text_caption(pretrained='../checkpoints/tag2text_swin_14m.pth',
                                    image_size=384,
                                    vit='swin_b',
                                    delete_tag_index=delete_tag_index,
                                    threshold = self.threshold)
        model.eval()
        # model = optimize_model(model)
        self.model = model
    
    def image_caption_detect_from_dir(self, image_dir):
        raw_image = Image.open(image_dir).convert("RGB")
        image = self.transform(raw_image).unsqueeze(0)
        print("Extract the image tag and caption results.")
        with torch.inference_mode():
            caption, tag_predict = self.model.generate(image,
                                                tag_input=None,
                                                max_length=50,
                                                return_tag_predict=True)
        print("Finished.")
        return tag_predict[0].replace(' | ', ', '), caption[0]
    
    def image_caption_detect_from_array(self, image):
        image = Image.fromarray(image)
        image = self.transform(image).unsqueeze(0)
        print("Extract the image tag and caption results.")
        with torch.inference_mode():
            caption, tag_predict = self.model.generate(image,
                                                tag_input=None,
                                                max_length=50,
                                                return_tag_predict=True)
        print("Finished.")
        return tag_predict[0].replace(' | ', ', '), caption[0]
    
    def image_caption_detect(self, image):
        image = self.transform(image).unsqueeze(0)
        print("Extract the image tag and caption results.")
        with torch.inference_mode():
            caption, tag_predict = self.model.generate(image,
                                                tag_input=None,
                                                max_length=50,
                                                return_tag_predict=True)
        print("Finished.")
        return tag_predict[0].replace(' | ', ', '), caption[0]