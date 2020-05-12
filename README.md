# Implementation of a DCGAN and a cycleGAN

The aims of this project are to implement a DCGAN and a cycleGAN.

We decided to apply the DCGAN to generate anime style face.  

We decided to apply the cycleGAN to transform real photographies to Monet's painting and to transform Monet's painting to real photographies.  
We also tried to apply the cycleGAN to transform MNIST dataset to USPS dataset and vive versa.  

Hardware = NVIDIA RTX 2070

# DCGAN

After 260 epochs with batch size equal to 36.

![DCGAN](https://github.com/YannickLy/DeepLearning-Project-ENSAE-2020/raw/master/Implementation/DCGAN/.py/images/epoch%20260.png)  

# cycleGAN

First row = real image from the domain X  
Second row = fake image from the domain X to domain Y  
Third row = real image fror the domain Y  
Fourth row = fake image from the domain Y to domain X  

### Monet2Photo
After 11 epochs with batch size equal to 1.

![cycleGAN monet2photo](https://github.com/YannickLy/DeepLearning-Project-ENSAE-2020/raw/master/Implementation/cycleGAN/monet2photo/.py/images/epoch%2011.png)  

### MNIST2USPS
After 13 epochs with batch size equal to 32.

![cycleGAN MNIST2USPS](https://github.com/YannickLy/DeepLearning-Project-ENSAE-2020/raw/master/Implementation/cycleGAN/MNIST2USPS/.py/images/epoch%2013.png)  

## Authors

* Jing Tan
* Yannick Ly
