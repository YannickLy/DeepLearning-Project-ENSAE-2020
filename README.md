# Implementation of a DCGAN and a cycleGAN

This is a homework for the course of Deep Learning ENSAE 2020 Spring. Find the instructions on: https://marcocuturi.net/dl.html

The aims of this project are to implement a DCGAN and a cycleGAN.

In the first part of the project, we used DCGAN to generate anime style face.  

Next, we decided to apply the cycleGAN to transform real photographies to Monet's painting and to transform Monet's painting to real photographies.  
We also tried to carry out a cycleGAN to transform MNIST dataset to USPS dataset and vice versa.  

Hardware GPU = NVIDIA RTX 2070

# DCGAN

After 260 epochs with batch size equal to 36 (~5 hours running).

![DCGAN](https://github.com/YannickLy/DeepLearning-Project-ENSAE-2020/raw/master/Implementation/DCGAN/.py/images/epoch%20260.png)  

# cycleGAN

First row = real image from the domain X  
Second row = fake image from the domain X transform to domain Y  
Third row = real image from the domain Y  
Fourth row = fake image from the domain Y transform to domain X

### Monet2Photo
After 11 epochs with batch size equal to 1 (~10 hours running).

![cycleGAN monet2photo](https://github.com/YannickLy/DeepLearning-Project-ENSAE-2020/raw/master/Implementation/cycleGAN/monet2photo/.py/images/epoch%2011.png)  

### MNIST2USPS
After 13 epochs with batch size equal to 32 (~5 hours running).

![cycleGAN MNIST2USPS](https://github.com/YannickLy/DeepLearning-Project-ENSAE-2020/raw/master/Implementation/cycleGAN/MNIST2USPS/.py/images/epoch%2013.png)  

## Authors

* Jing Tan
* Yannick Ly
