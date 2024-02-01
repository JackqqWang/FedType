
import torch.nn as nn
import torch.nn.functional as F


from options import args_parser
args = args_parser()


class VGG(nn.Module):
    
    def __init__(self, features, output_dim):
        super().__init__()

        self.features = features

        self.avgpool = nn.AdaptiveAvgPool2d(7)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x


vgg11_config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

vgg13_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512,
                512, 'M']

vgg16_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512,
                'M', 512, 512, 512, 'M']

vgg19_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512,
                512, 512, 'M', 512, 512, 512, 512, 'M']

def get_vgg_layers(config, batch_norm):

    layers = []
    in_channels = 3

    for c in config:
        assert c == 'M' or isinstance(c, int)
        if c == 'M':
            layers += [nn.MaxPool2d(kernel_size=2)]
        else:
            conv2d = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = c

    return nn.Sequential(*layers)



def vgg11(OUTPUT_DIM = args.num_classes):
    vgg11_layers = get_vgg_layers(vgg11_config, batch_norm=True)
    model = VGG(vgg11_layers, OUTPUT_DIM)
    return model



def vgg13(OUTPUT_DIM = args.num_classes):
    vgg13_layers = get_vgg_layers(vgg13_config, batch_norm=True)
    model = VGG(vgg13_layers, OUTPUT_DIM)
    return model


def vgg16(OUTPUT_DIM = args.num_classes):
    vgg16_layers = get_vgg_layers(vgg16_config, batch_norm=True)
    model = VGG(vgg16_layers, OUTPUT_DIM)
    return model


def vgg19(OUTPUT_DIM = args.num_classes):
    vgg19_layers = get_vgg_layers(vgg19_config, batch_norm=True)
    model = VGG(vgg19_layers, OUTPUT_DIM)
    return model

