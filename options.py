import argparse
import torch

def args_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--core_model', type=str, default='resnet18')
    parser.add_argument('--pre_trained', type=int, default= 0, help="1 is pretrained, 0 is not pretrained")
    parser.add_argument('--ablation', type=int, default= 0, help="1 kd")
    parser.add_argument('--fine_tune_epoch', type=int, default= 5, help="num of fine tune epoch")
    parser.add_argument('--small_kreg', type=int, default= 5, help="1, 3, 5, 7")
    parser.add_argument('--small_lamda', type=float, default=0.5,
                        help='0.05, 0.1, 0.5, 1')
    parser.add_argument('--large_kreg', type=int, default= 5, help="1, 3, 5, 7")
    parser.add_argument('--large_lamda', type=float, default=1,
                        help='0.05, 0.1, 0.5, 1')
    parser.add_argument('--lambda_function', type=str, default='fenduan_linear'\
                        'help: linear\
                            fenduan_linear\
                            fenduan_nonliner')

    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dataset', type=str, default='cifar10', help="name \
                        of dataset,cifar10, cifar100, fmnist, ISIC ")

    # dataset parameters
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes, CIFAR100: 100, CIFAR10: 10")
    
    # federated parameters
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K")
    parser.add_argument('--communication_round', type=int, default=200, help="number \
                        of communication round")
    parser.add_argument('--frac', type=float, default=0.2,
                        help='the fraction of clients: C')
    parser.add_argument('--iid', type=int, default=0,
                        help='1 is iid, 0 is non-iid')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    
    # local parameters
    parser.add_argument('--local_bs', type=int, default=16,
                        help="local batch size: B")
    parser.add_argument('--local_ep', type=int, default=5,
                        help="the number of local epochs: E")
    parser.add_argument('--temperature', type=float, default=1,
                        help='T in KD') 
    parser.add_argument('--optimizer', type=str, default='adam', help="type \
                        of optimizer")
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    args = parser.parse_args()

    return args
