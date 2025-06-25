from .clip import clip 
from PIL import Image
import torch.nn as nn


CHANNELS = {
    "RN50" : 1024,
    "ViT-L/14" : 768
}

class CLIPModel(nn.Module):
    
    def __init__(self, name, design_details, num_classes=256):
        super(CLIPModel, self).__init__()

        self.model, self.preprocess = clip.load(name, design_details, device="cpu") # self.preprecess will not be used during training, which is handled in Dataset class 
        #self.fc = nn.Linear( CHANNELS[name], num_classes )
 

    def forward(self, x, prompt_ids, return_feature=False):
        features = self.model.encode_image(x, prompt_ids) 
        #if return_feature:
        #    return features
        #out = self.fc(features)
        return features

