import os
import sys
import wget
import zipfile
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import argparse

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import datasets

from architectures import *
from helper import *

##############################
#        GET DATA            # 
##############################

url = "https://www.dropbox.com/s/a0ypfj5rvlxux7l/mangacharac.zip?dl=1"
wget.download(url, "../../../Data/")

with zipfile.ZipFile("../../../Data/mangacharac.zip", 'r') as zip_ref:
    zip_ref.extractall("../../../Data/")

##############################
#    MODEL CONFIGURATION     # 
##############################

parser = argparse.ArgumentParser()
# data set
parser.add_argument("--dataset_name", type=str, default="mangacharac", help="name of the dataset")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
# DCGAN parameters
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.3, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=15, help="epoch from which to start lr decay")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
# training parameters
parser.add_argument("--cuda", type=bool, default=False, help="change to GPU mode")
parser.add_argument("--n_epochs", type=int, default=1000, help = "number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation") # 0 by default in windows because multiprocessing doesn't work
# saving parameters
parser.add_argument("--epoch", type=int, default=358, help = "epoch to start training from")
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

# Configure dataloader
transforms_= [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]

dataloader = DataLoader(
    ImageDataset("../../../Data/%s" % opt.dataset_name, transforms_ = transforms_),
    batch_size = opt.batch_size,
    shuffle = True,
    num_workers = opt.n_cpu
)

##############################
#    MODEL INITIALIZATION    # 
##############################

def create_model(opt):
    
    """ Builds the generators and discriminators. """
    
    # Initialize generator and discriminator
    G = Generator(opt)
    D = Discriminator(opt)
    
    print_models(G, D)
    
    if opt.cuda:
        G = G.cuda()
        D = D.cuda()
        
    if opt.epoch != 0:
        # Load pretrained models
        G.load_state_dict(torch.load("saved_models/G_%d.pth" % opt.epoch))
        D.load_state_dict(torch.load("saved_models/D_%d.pth" % opt.epoch))
    else:
        # Initialize weights
        G.apply(init_normal_weights)
        D.apply(init_normal_weights)
        
    return G, D

##############################
#       MODEL TRAINING       # 
##############################
    
def training_loop(dataloader, opt):
    
    """ Runs the training loop.
        * Saves checkpoint every opt.checkpoint_interval iterations
        * Saves generated samples every opt.sample_interval iterations
    """
    
    # Create generators and discriminators
    G, D = create_model(opt)
    
    # Loss
    loss_GAN = torch.nn.BCELoss()

    # Optimizers
    optimizer_G = torch.optim.Adam(G.parameters(), lr = opt.lr, betas = (opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(D.parameters(), lr = opt.lr, betas = (opt.b1, opt.b2))

    # Learning rate update schedulers
    LambdaLR_schedular_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda = LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    LambdaLR_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda = LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    
    Tensor = torch.Tensor
    if opt.cuda:  
        loss_GAN.cuda()
        Tensor = torch.cuda.FloatTensor

    losses_models_G = pd.DataFrame(np.zeros((len(dataloader), opt.n_epochs - opt.epoch + 1)))
    losses_models_D = pd.DataFrame(np.zeros((len(dataloader), opt.n_epochs - opt.epoch + 1)))
    losses_models_G.columns = range(opt.epoch, opt.n_epochs+1)
    losses_models_D.columns = range(opt.epoch, opt.n_epochs+1)
    # Training
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):
            
            # Configure input
            real_imgs = Variable(batch.type(Tensor))
            
            # Adversarial ground truths
            valid = Variable(Tensor(batch.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(batch.shape[0], 1).fill_(0.0), requires_grad=False)
            
            # ------------------
            #  Train Generator 
            # ------------------

            optimizer_G.zero_grad()
            
            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (batch.shape[0], opt.latent_dim))))

            # Generate a batch of images
            gen_imgs = G(z)

            # Generator loss
            loss_G = loss_GAN(D(gen_imgs), valid)
    
            loss_G.backward()
            optimizer_G.step()
            
            losses_models_G[epoch][i] = loss_G
                        
            # --------------------
            #  Train Discriminator 
            # --------------------
            
            optimizer_D.zero_grad()

            # Discriminator loss
            real_loss = loss_GAN(D(real_imgs), valid)
            fake_loss = loss_GAN(D(gen_imgs.detach()), fake)
            loss_D = (real_loss + fake_loss) / 2
    
            loss_D.backward()
            optimizer_D.step()
            
            losses_models_D[epoch][i] = loss_D
           
            # --------------
            #  Log Progress
            # --------------
        
            batches_done = epoch * len(dataloader) + i
        
            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item()
                )
            )
        
            # Save sample image at interval
            if batches_done % opt.sample_interval == 0:
                save_image(gen_imgs.data[:32], "images/%d.png" % batches_done, nrow=4, normalize=True)
        
        # Update learning rates
        LambdaLR_schedular_G.step()
        LambdaLR_scheduler_D.step()
        
        # Save discriminators and generators lossses at each epoch
        losses_models_G.to_pickle("losses_models/losses_models_G_%d" % epoch)
        losses_models_D.to_pickle("losses_models/losses_models_D_%d" % epoch)
            
        # Save model at checkpoints
        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            torch.save(G.state_dict(), "saved_models/G_%d.pth" % epoch)
            torch.save(D.state_dict(), "saved_models/D_%d.pth" % epoch)
            
training_loop(dataloader, opt)