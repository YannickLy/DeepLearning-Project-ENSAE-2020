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

def print_models(G_XtoY, G_YtoX, D_X, D_Y):
    
    """ Prints model information for the generators and discriminators. """
    print("                 G_XtoY                ")
    print("---------------------------------------")
    print(G_XtoY)
    print("---------------------------------------")

    print("                 G_YtoX                ")
    print("---------------------------------------")
    print(G_YtoX)
    print("---------------------------------------")

    print("                  D_X                  ")
    print("---------------------------------------")
    print(D_X)
    print("---------------------------------------")

    print("                  D_Y                  ")
    print("---------------------------------------")
    print(D_Y)
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
    
    def __init__(self, root, transforms_, mode):
        self.transform = transforms.Compose(transforms_)
        self.files_X = sorted(glob.glob(os.path.join(root, "%sA" % mode) + "/*.*"))
        self.files_Y = sorted(glob.glob(os.path.join(root, "%sB" % mode) + "/*.*"))
        
    def __to_rgb(self, image):
        rgb_image = Image.new("RGB", image.size)
        rgb_image.paste(image)
        return rgb_image

    def __getitem__(self, index):
        image_X = Image.open(self.files_X[index % len(self.files_X)])
        image_Y = Image.open(self.files_Y[index % len(self.files_Y)])
        if image_X.mode != "RGB":
            image_X = self.__to_rgb(image_X)
        if image_Y.mode != "RGB":
            image_Y = self.__to_rgb(image_Y)
        item_X = self.transform(image_X)
        item_Y = self.transform(image_Y)
        return {"X": item_X, "Y": item_Y}
    
    def __len__(self):
        return max(len(self.files_X), len(self.files_Y))


