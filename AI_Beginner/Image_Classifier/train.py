import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from collections import OrderedDict
import torchvision
from torchvision import datasets, transforms, models

import os
import time
import json
import copy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', action='store')
parser.add_argument('--save_dir', dest='save_dir')
parser.add_argument('--arch', dest='arch')
parser.add_argument('--learning_rate', type=float)
parser.add_argument('--hidden_units', type=int)
parser.add_argument('--epochs', type=int)
parser.add_argument('--gpu', action='store_true')

try:
    args = parser.parse_args()
except IOError as msg:
    parser.error(str(msg))
    exit(0)


def build_model(arch, hidden_units):
    if arch == 'vgg19':
        model = models.vgg19(pretrained=True)
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        model = None

    if model is not None:
        for param in model.features.parameters():
            param.requires_grad = False

        # save number of features of last layer
        num_features = model.classifier[0].in_features
        classifier = nn.Sequential(OrderedDict([
                        ('fc1', nn.Linear(num_features, hidden_units)),
                        ('relu', nn.ReLU()),
                        ('dropout', nn.Dropout(0.5)),
                        ('fc2', nn.Linear(hidden_units, 102)),
                        ('output', nn.LogSoftmax(dim=1))
                        ]))
        model.classifier = classifier

    print(model)
    return model


def build_transforms():
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]),
        'valid': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]),
        'test': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    }

    return data_transforms


def build_datasets(data_dir):
    data_transforms = build_transforms()
    return {
        stage: datasets.ImageFolder(
            os.path.join(data_dir, stage),
            transform=data_transforms[stage]
        )
        for stage in ['train', 'valid', 'test']
    }


def build_dataloaders(image_datasets):
    return {
        stage: torch.utils.data.DataLoader(
            image_datasets[stage],
            batch_size=32,
            shuffle=True
        )
        for stage in ['train', 'valid', 'test']
    }


def train_model(data_dir, arch, epochs, learning_rate, hidden_units, use_gpu=True):
    if not use_gpu:
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            print("there is no gpu available, use cpu instead")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:0")

    model = build_model(arch, hidden_units)

    #print(model)

    if model is None:
        print('Unknown model, please check your model name and try again')
        return

    if arch == 'vgg19' or arch == 'vgg16':
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate, momentum=0.9)
        sched = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    elif arch == 'densenet121':
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adadelta(model.parameters())
        sched = lr_scheduler.StepLR(optimizer, step_size=4)
    else:
        pass

    image_datasets = build_datasets(data_dir)

    data_loaders = build_dataloaders(image_datasets)

    dataset_sizes = {stage: len(image_datasets[stage])
                     for stage in ['train', 'valid']}

    model = model.to(device)
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    avg_loss = 0
    avg_acc = 0
    avg_loss_val = 0
    avg_acc_val = 0

    train_batches = len(data_loaders['train'])
    valid_batches = len(data_loaders['valid'])

    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch, epochs))
        print('-' * 10)

        loss_train = 0
        loss_val = 0
        acc_train = 0
        acc_val = 0

        # set model to training state
        model.train(True)

        for i, data in enumerate(data_loaders['train']):
            if i % 10 == 0:
                print("\rTraining batch {}/{}".format(i,
                                                      train_batches), end='', flush=True)

            if i >= train_batches:
                break

            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            _, predicts = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            acc_train += torch.sum(predicts == labels.data)

            del inputs, labels, outputs, predicts
            torch.cuda.empty_cache()

        print()

        print("acc_train: {:.4f}".format(acc_train))
        print("size of train data: {:.4f}".format(dataset_sizes['train']))
        avg_loss = loss_train / dataset_sizes['train']
        avg_acc = acc_train.item() / dataset_sizes['train']

        # set model to evaluation state
        model.train(False)
        model.eval()

        for i, data in enumerate(data_loaders['valid']):
            if i % 10 == 0:
                print("\rValidation batch {}/{}".format(i,
                                                        valid_batches), end='', flush=True)

            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            _, predicts = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            loss_val += loss.item()
            acc_val += torch.sum(predicts == labels.data)

            del inputs, labels, outputs, predicts
            torch.cuda.empty_cache()

        print("acc_val: {:.4f}".format(acc_val))
        print("size of train data: {:.4f}".format(dataset_sizes['valid']))
        avg_loss_val = loss_val / dataset_sizes['valid']
        avg_acc_val = acc_val.item() / dataset_sizes['valid']

        print()
        print("Epoch {} result: ".format(epoch))
        print("Avg loss (train): {:.4f}".format(avg_loss))
        print("Avg acc (train): {:.4f}".format(avg_acc))
        print("Avg loss (val): {:.4f}".format(avg_loss_val))
        print("Avg acc (val): {:.4f}".format(avg_acc_val))
        print('-' * 10)
        print()

        if avg_acc_val > best_acc:
            best_acc = avg_acc_val
            best_model_wts = copy.deepcopy(model.state_dict())

    elapsed_time = time.time() - since
    print()
    print("Training completed in {:.0f}m {:.0f}s".format(
        elapsed_time // 60, elapsed_time % 60))
    print("Best acc: {:.4f}".format(best_acc))

    model.load_state_dict(best_model_wts)
    class_to_idx = image_datasets['train'].class_to_idx
    return model, criterion, device, class_to_idx


def eval_model(model, data_dir, criterion, device):
    since = time.time()
    avg_loss = 0
    avg_acc = 0
    loss_test = 0
    acc_test = 0

    image_datasets = build_datasets(data_dir)

    data_loaders = build_dataloaders(image_datasets)

    dataset_sizes = {stage: len(image_datasets[stage])
                     for stage in ['test']}
    test_batches = len(data_loaders['test'])
    print("Evaluating model")
    print('-' * 10)

    for i, data in enumerate(data_loaders['test']):
        if i % 10 == 0:
            print("\rTest batch {}/{}".format(i, test_batches), end='', flush=True)

        # set model for evaluation state
        model.train(False)
        model.eval()

        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        _, predicts = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        loss_test += loss.item()
        acc_test += torch.sum(predicts == labels.data)

        del inputs, labels, outputs, predicts
        torch.cuda.empty_cache()

    avg_loss = loss_test / dataset_sizes['test']
    avg_acc = acc_test.item() / dataset_sizes['test']

    print("acc (test): {:.4f}".format(acc_test))
    print("size of (test): {:.4f}".format(dataset_sizes['test']))
    print("Avg acc (test): {:.4f}".format(avg_acc))
    elapsed_time = time.time() - since
    print()
    print("Evaluation completed in {:.0f}m {:.0f}s".format(
        elapsed_time // 60, elapsed_time % 60))
    print("Avg loss (test): {:.4f}".format(avg_loss))
    print("Avg acc (test): {:.4f}".format(avg_acc))
    print('-' * 10)


data_dir = './flowers/' if args.data_dir is None else args.data_dir
save_dir = './' if args.save_dir is None else args.save_dir
arch = 'vgg16' if args.arch is None else args.arch
learning_rate = 0.001 if args.learning_rate is None else args.learning_rate
hidden_units = 512 if args.hidden_units is None else args.hidden_units
epochs = 1 if args.epochs is None else args.epochs
gpu = args.gpu

trained_model, criterion, device, class_to_idx = train_model(data_dir, arch, epochs, learning_rate, hidden_units, gpu)

print(trained_model)

eval_model(trained_model, data_dir, criterion, device)

checkpoint = {'model': trained_model,
              'state_dict': trained_model.state_dict(),
              'class_to_idx': class_to_idx
              }

torch.save(checkpoint, os.path.join(save_dir, 'checkpoint_{}.pth'.format(arch)))