import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch
from options import args_parser
args = args_parser()


resnet18 = models.resnet18(pretrained=args.pre_trained)
reset34 = models.resnet34(pretrained=args.pre_trained)
reset50 = models.resnet50(pretrained=args.pre_trained)
resenet101 = models.resnet101(pretrained=args.pre_trained)
resnet152 = models.resnet152(pretrained=args.pre_trained)
vgg11 = models.vgg11(pretrained=args.pre_trained)
vgg13 = models.vgg13(pretrained=args.pre_trained)
vgg16 = models.vgg16(pretrained=args.pre_trained)
vgg19 = models.vgg19(pretrained=args.pre_trained)


class CustomResNet:
    def __init__(self, model, name):
        self.model = model
        self.name = name

    def __getattr__(self, attr):
        return getattr(self.model, attr)

resnet18 = CustomResNet(models.resnet18(pretrained=args.pre_trained), 'resnet18')
reset34 = CustomResNet(models.resnet34(pretrained=args.pre_trained), 'resnet34')
reset50 = CustomResNet(models.resnet50(pretrained=args.pre_trained), 'resnet50')
resenet101 = CustomResNet(models.resnet101(pretrained=args.pre_trained), 'resnet101')
resnet152 = CustomResNet(models.resnet152(pretrained=args.pre_trained), 'resnet152')

# model_list = [resnet18, reset34, reset50, resenet101, resnet152, vgg11, vgg13, vgg16, vgg19]
resent_model_list = [resnet18, reset34, reset50, resenet101, resnet152]
vgg_model_list = [vgg11, vgg13, vgg16, vgg19]

def resnet_modify_function(input_list, num_classes):
    for model in input_list:
        num_ftrs = model.model.fc.in_features
        model.model.fc = nn.Linear(num_ftrs, num_classes)
    return input_list

def vgg_modify_function(input_list, num_classes):
    for model in input_list:
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, num_classes)
    return input_list



resent_model_modify_list = resnet_modify_function(resent_model_list, num_classes = args.num_classes)









