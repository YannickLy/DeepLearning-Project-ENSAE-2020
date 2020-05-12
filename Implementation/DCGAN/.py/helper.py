import glob
import os
from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

def init_normal_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def print_models(G, D):
    
    """ Prints model information for the generators and discriminators. """
    
    print("                   G                   ")
    print("---------------------------------------")
    print(G)
    print("---------------------------------------")

    print("                   D                   ")
    print("---------------------------------------")
    print(D)
    print("---------------------------------------")
    
class LambdaLR:
    
    """ LambdaLR or LambdaLearningRate
        Allow us to decrease the learning rate from a specific epoch ('decay_start_epoch')
        This accelerates learning and become and allows to be more precise for larger epochs.
    """
    
    def __init__(self, n_epochs, offset, decay_start_epoch):
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch
        
    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)
    
class ImageDataset(Dataset):
    
    """ ImageDataSet
        Allow us to preprocess/transform data into a desired format (resize, crop, normalize, rgb, tensor)
    """
    
    def __init__(self, root, transforms_):
        self.transform = transforms.Compose(transforms_)
        self.files = sorted(glob.glob(os.path.join(root) + "/*.*"))

    def __getitem__(self, index):
        image = Image.open(self.files[index % len(self.files)])
        image = self.transform(image)
        return image
    
    def __len__(self):
        return len(self.files)



