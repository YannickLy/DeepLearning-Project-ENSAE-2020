import torch.nn as nn

class Generator(nn.Module):

    """Defines the architecture of the generator network."""
    
    def __init__(self, opt):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.linear = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))
        
        out_features = 128
        model = [nn.BatchNorm2d(out_features)]
        in_features = out_features
        for _ in range(2):
            out_features //= 2
            model += [
                    nn.Upsample(scale_factor = 2),
                    nn.Conv2d(in_features, out_features, 3, stride = 1, padding = 1),
                    nn.BatchNorm2d(out_features, 0.8),
                    nn.LeakyReLU(0.2, inplace = True)
                    ]
            in_features = out_features
        model += [nn.Conv2d(out_features, 3, opt.channels, stride = 1, padding = 1), nn.Tanh()]
        
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.shape[0], 128, self.init_size, self.init_size)
        x = self.model(x)
        return x

class Discriminator(nn.Module):
    
    """Defines the architecture of the discriminator network."""
    
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, batch_normalize = True):
            block = [nn.Conv2d(in_filters, out_filters, 3, stride = 2, padding = 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if batch_normalize:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, batch_normalize = False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 4 ** 2
        self.output_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.shape[0], -1)
        x = self.output_layer(x)
        return x