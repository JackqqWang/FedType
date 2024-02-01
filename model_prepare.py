from model_pool_resnet import *
from model_pool_vgg import *
from mobilenetv1 import *
from mobilenetv2 import *
from mobilenetv3 import *
from support_tools import *
from options import args_parser
import torchvision.models as models
import torchvision
from resnet_vgg_pool_torch import *
args = args_parser()
device = args.device

if args.pre_trained:
    model = torchvision.models.resnet18(pretrained=True,progress=True).to(device)
else:
    model = torchvision.models.resnet18(pretrained=False,progress=True).to(device)


model_pool_list = resent_model_modify_list



def list_to_dict(input_list):

    output_dict = {}
    output_dict = {index: item for index, item in enumerate(input_list)}
    return output_dict

output_dict = list_to_dict(model_pool_list)

