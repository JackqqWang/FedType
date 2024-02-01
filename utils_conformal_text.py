import copy
import json
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset, DatasetDict
from torch.utils.data import DataLoader, Subset
from transformers import BertTokenizer, DistilBertTokenizer
from torch.optim import AdamW, SGD, Adam
from sampling import iid
from sampling import sst2_noniid, ag_news_noniid, text_noniid_dirichlet
from sampling import cifar_iid, cifar_noniid, cifar_noniid_dirichlet
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from tqdm import tqdm
from torch.nn.functional import kl_div, softmax, log_softmax, cross_entropy
from collections import OrderedDict
import functools


def get_tokenizer(args):

    if args.model == 'bert':
        # Load the BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif args.model == 'distill_bert':
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    else:
        exit(f'Error: no {args.model} model')

    return tokenizer


def tokenize_dataset(args, dataset):
    text_field_key = 'text' if args.dataset == 'ag_news' else 'sentence'
    tokenizer = get_tokenizer(args)

    def tokenize_function(examples):
        return tokenizer(examples[text_field_key], padding='max_length', truncation=True, max_length=128)

    # tokenize the training and test set
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.with_format("torch")

    return tokenized_dataset


def get_dataset(args):
    # text_field_key = 'text' if args.dataset == 'ag_news' else 'sentence'
    val_key = 'test' if args.dataset == 'ag_news' else 'validation'

    # load dataset
    if args.dataset == 'sst2':
        dataset = load_dataset('glue', args.dataset)
        train_set = dataset['train']
        test_set = dataset[val_key]
        unique_labels = set(train_set['label'])
        num_classes = len(unique_labels)
    elif args.dataset == 'ag_news':
        dataset = load_dataset("ag_news")
        train_set = dataset['train']
        test_set = dataset[val_key]
        unique_labels = set(train_set['label'])
        num_classes = len(unique_labels)
    elif args.dataset == 'cifar10':
        data_dir = './data/cifar10/'
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        train_set = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
        num_classes = 10
    elif args.dataset == 'cifar100':
        data_dir = './data/cifar100/'
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        train_set = datasets.CIFAR100(data_dir, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR100(data_dir, train=False, download=True, transform=transform)
        num_classes = 100
    else:
        exit(f'Error: no {args.dataset} dataset')

    if args.iid:
        if args.dataset == 'cifar10' or 'cifar100':
            user_groups = cifar_iid(train_set, args.num_users)
        else:
            user_groups = iid(train_set, args.num_users)
    else:
        if args.dataset == 'sst2':
            user_groups = sst2_noniid(train_set, args.num_users)
        elif args.dataset == 'ag_news':
            # user_groups = ag_news_noniid(train_set, args.num_users)
            user_groups = text_noniid_dirichlet(train_set, args.num_users, args.beta)
        elif args.dataset == 'cifar10' or 'cifar100':
            # user_groups = cifar_noniid(train_set, args.num_users)
            user_groups = cifar_noniid_dirichlet(train_set, args.num_users, args.beta)
        else:
            exit(f'Error: non iid split is not implemented for the {args.dataset} dataset')

    return train_set, test_set, num_classes, user_groups

