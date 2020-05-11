import os
import sys
import wget
import zipfile
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import itertools
import argparse

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image

from architectures import *
from helper import *

##############################
#        GET DATA            # 
##############################

url = "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/monet2photo.zip"
wget.download(url, "../../../../Data/")

with zipfile.ZipFile("../../../../Data/monet2photo.zip", 'r') as zip_ref:
    zip_ref.extractall("../../../../Data/")


##############################
#    MODEL CONFIGURATION     # 
##############################

parser = argparse.ArgumentParser()
# data set
parser.add_argument("--dataset_name", type=str, default="monet2photo", help="name of the dataset")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
# cycleGAN parameters
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.3, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=10, help="epoch from which to start lr decay")
# training parameters
parser.add_argument("--cuda", type=bool, default=False, help="change to GPU mode")
parser.add_argument("--n_epochs", type=int, default=50, help = "number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation") # 0 by default in windows because multiprocessing doesn't work
# saving parameters
parser.add_argument("--epoch", type=int, default=0, help = "epoch to start training from")
parser.add_argument("--sample_interval", type=int, default=1000, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model checkpoints")
opt = parser.parse_args()

# GPU option
cuda = torch.cuda.is_available()
if cuda:
    print('Models moved to GPU.')
    opt.cuda = True

# Create sample image, checkpoint and losses directories
os.makedirs("images", exist_ok = True)
os.makedirs("saved_models", exist_ok = True)
os.makedirs("losses_models", exist_ok = True)

# Image transformations
transforms_ = [
    transforms.Resize(int(opt.img_height * 1.12), Image.BICUBIC),
    # Add some nose into the data
    transforms.RandomCrop((opt.img_height, opt.img_width)),
    # Random horizontal flip on image
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # Normalise data
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

# Train dataloader
train_dataloader = DataLoader(
    ImageDataset("../../../../Data/%s" % opt.dataset_name, transforms_ = transforms_, mode = "train"),
    batch_size = opt.batch_size,
    shuffle = True,
    num_workers = opt.n_cpu,
)

# Test dataloader
test_dataloader = DataLoader(
    ImageDataset("../../../../Data/%s" % opt.dataset_name, transforms_= transforms_, mode = "test"),
    batch_size = 5,
    shuffle = True,
    num_workers = opt.n_cpu,
)

##############################
#    MODEL INITIALIZATION    # 
##############################

def create_model(opt):
    
    """ Builds the generators and discriminators. """
    
    input_shape = (opt.channels, opt.img_height, opt.img_width)
    
    # Initialize generator and discriminator
    G_XtoY = Generator(input_shape, opt.n_residual_blocks)
    G_YtoX = Generator(input_shape, opt.n_residual_blocks)
    D_X = Discriminator(input_shape)
    D_Y = Discriminator(input_shape)
    
    print_models(G_XtoY, G_YtoX, D_X, D_Y)
    
    if opt.cuda:
        G_XtoY = G_XtoY.cuda()
        G_YtoX = G_YtoX.cuda()
        D_X = D_X.cuda()
        D_Y = D_Y.cuda()
        
    if opt.epoch != 0:
        # Load pretrained models
        G_XtoY.load_state_dict(torch.load("saved_models/G_XtoY_%d.pth" % opt.epoch))
        G_YtoX.load_state_dict(torch.load("saved_models/G_YtoX_%d.pth" % opt.epoch))
        D_X.load_state_dict(torch.load("saved_models/D_X_%d.pth" % opt.epoch))
        D_Y.load_state_dict(torch.load("saved_models/D_Y_%d.pth" % opt.epoch))
    else:
        # Initialize weights
        G_XtoY.apply(init_normal_weights)
        G_YtoX.apply(init_normal_weights)
        D_X.apply(init_normal_weights)
        D_Y.apply(init_normal_weights)
        
    return G_XtoY, G_YtoX, D_X, D_Y

##############################
#       MODEL TRAINING       # 
##############################
    
def training_loop(train_dataloader, test_dataloader, opt):
    
    """ Runs the training loop.
        * Saves checkpoint every opt.checkpoint_interval iterations
        * Saves generated samples every opt.sample_interval iterations
    """
    
    # Create generators and discriminators
    G_XtoY, G_YtoX, D_X, D_Y = create_model(opt)
    
    # Losses
    loss_GAN = torch.nn.MSELoss()
    loss_cycle = torch.nn.L1Loss()
    loss_identity = torch.nn.L1Loss()

    # Optimizers
    optimizer_G = torch.optim.Adam(itertools.chain(G_XtoY.parameters(), G_YtoX.parameters()), lr = opt.lr, betas = (opt.b1, opt.b2))
    optimizer_D_X = torch.optim.Adam(D_X.parameters(), lr = opt.lr, betas = (opt.b1, opt.b2))
    optimizer_D_Y = torch.optim.Adam(D_Y.parameters(), lr = opt.lr, betas = (opt.b1, opt.b2))

    # Learning rate update schedulers
    LambdaLR_schedular_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda = LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    LambdaLR_scheduler_D_X = torch.optim.lr_scheduler.LambdaLR(optimizer_D_X, lr_lambda = LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    LambdaLR_scheduler_D_Y = torch.optim.lr_scheduler.LambdaLR(optimizer_D_Y, lr_lambda = LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    
    Tensor = torch.Tensor
    if opt.cuda:  
        loss_GAN.cuda()
        loss_cycle.cuda()
        loss_identity.cuda()
        Tensor = torch.cuda.FloatTensor

    def sample_images(batches_done):
        """Saves a generated sample from the test set"""
        imgs = next(iter(test_dataloader))
        G_XtoY.eval()
        G_YtoX.eval()
        real_X = Variable(imgs["X"].type(Tensor))
        fake_Y = G_XtoY(real_X)
        real_Y = Variable(imgs["Y"].type(Tensor))
        fake_X = G_YtoX(real_Y)
        # Arange images along x-axis
        real_X = make_grid(real_X, nrow = 5, normalize = True)
        real_Y = make_grid(real_Y, nrow = 5, normalize = True)
        fake_X = make_grid(fake_X, nrow = 5, normalize = True)
        fake_Y = make_grid(fake_Y, nrow = 5, normalize = True)
        # Arange images along y-axis
        image_grid = torch.cat((real_X, fake_Y, real_Y, fake_X), 1)
        save_image(image_grid, "images/%s.png" % batches_done, normalize = False)
    
    losses_models_G = pd.DataFrame(np.zeros((len(train_dataloader), opt.n_epochs - opt.epoch + 1)))
    losses_models_D = pd.DataFrame(np.zeros((len(train_dataloader), opt.n_epochs - opt.epoch + 1)))
    losses_models_G.columns = range(opt.epoch, opt.n_epochs+1)
    losses_models_D.columns = range(opt.epoch, opt.n_epochs+1)
    # Training
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(train_dataloader):
            
            # Set model input
            real_X = Variable(batch["X"].type(Tensor))
            real_Y = Variable(batch["Y"].type(Tensor))
            
            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((real_X.size(0), *D_X.output_shape))), requires_grad = False)
            fake = Variable(Tensor(np.zeros((real_X.size(0), *D_X.output_shape))), requires_grad = False)
                        
            # -----------------------
            #  Train Discriminator X
            # -----------------------
            
            optimizer_D_X.zero_grad()
        
            # Real loss
            loss_real_D_X = loss_GAN(D_X(real_X), valid)
            
            # Fake loss
            fake_X = G_YtoX(real_Y)
            loss_fake_D_X = loss_GAN(D_X(fake_X), fake)
            
            # Total loss
            loss_D_X = (loss_real_D_X + loss_fake_D_X) / 2
        
            loss_D_X.backward()
            optimizer_D_X.step()
            
            # -----------------------
            #  Train Discriminator Y
            # -----------------------
        
            optimizer_D_Y.zero_grad()
            
            # Real loss
            loss_real_D_Y = loss_GAN(D_Y(real_Y), valid)
            
            # Fake loss
            fake_Y = G_XtoY(real_X)
            loss_fake_D_Y = loss_GAN(D_Y(fake_Y), fake)
            
            # Total loss
            loss_D_Y = (loss_real_D_Y + loss_fake_D_Y) / 2
        
            loss_D_Y.backward()
            optimizer_D_Y.step()
        
            loss_D = (loss_D_X + loss_D_Y) / 2
            losses_models_D[epoch][i] = loss_D
            
            # ------------------------------
            #  Train Generators XtoY and YtoX
            # -------------------------------
            
            G_XtoY.train()
            G_YtoX.train()
            
            optimizer_G.zero_grad()
            
            # GAN loss
            fake_Y = G_XtoY(real_X)
            loss_GAN_XtoY = loss_GAN(D_Y(fake_Y), valid)
            fake_X = G_YtoX(real_Y)
            loss_GAN_YtoX = loss_GAN(D_X(fake_X), valid)
            loss_GAN_G = (loss_GAN_XtoY + loss_GAN_YtoX) / 2
            
            # Cycle loss
            recov_X = G_YtoX(fake_Y)
            loss_cycle_X = loss_cycle(recov_X, real_X)
            recov_Y = G_XtoY(fake_X)
            loss_cycle_Y = loss_cycle(recov_Y, real_Y)
            loss_cycle_G = (loss_cycle_X + loss_cycle_Y) / 2
            
            # Identity loss
            loss_identity_X = loss_identity(G_YtoX(real_X), real_X)
            loss_identity_Y = loss_identity(G_XtoY(real_Y), real_Y)
            loss_identity_G = (loss_identity_X + loss_identity_Y) / 2
            
            # Total loss
            loss_G = loss_GAN_G + opt.lambda_cyc * loss_cycle_G + opt.lambda_id * loss_identity_G
            
            loss_G.backward()
            optimizer_G.step()
            losses_models_G[epoch][i] = loss_G
        
            # --------------
            #  Log Progress
            # --------------
        
            batches_done = epoch * len(train_dataloader) + i
        
            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(train_dataloader),
                    loss_D.item(),
                    loss_G.item()
                )
            )
        
            # Save sample image at interval
            if batches_done % opt.sample_interval == 0:
                sample_images(batches_done)
        
        # Update learning rates
        LambdaLR_schedular_G.step()
        LambdaLR_scheduler_D_X.step()
        LambdaLR_scheduler_D_Y.step()
        
        # Save discriminators and generators lossses at each epoch
        losses_models_G.to_pickle("losses_models/losses_models_G_%d" % epoch)
        losses_models_D.to_pickle("losses_models/losses_models_D_%d" % epoch)
            
        # Save model at checkpoints
        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            torch.save(G_XtoY.state_dict(), "saved_models/G_XtoY_%d.pth" % epoch)
            torch.save(G_YtoX.state_dict(), "saved_models/G_YtoX_%d.pth" % epoch)
            torch.save(D_X.state_dict(), "saved_models/D_X_%d.pth" % epoch)
            torch.save(D_Y.state_dict(), "saved_models/D_Y_%d.pth" % epoch)
            
training_loop(train_dataloader, test_dataloader, opt)