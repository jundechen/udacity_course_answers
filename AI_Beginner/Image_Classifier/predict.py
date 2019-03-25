import argparse
import torch
from torch import nn
from torchvision import transforms, models
import json
from PIL import Image
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('input', action='store')
parser.add_argument('checkpoint', action='store')
parser.add_argument('--topk', type=int)
parser.add_argument('--category_names', type=str)
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--arch', dest='arch')

try:
    args = parser.parse_args()
except IOError as msg:
    parser.error(str(msg))
    exit(0)

print(args)

arch = 'vgg16' if args.arch is None else args.arch
category_names = 'cat_to_name.json' if args.category_names is None else args.category_names
checkpoint = 'checkpoint_vgg16.pth' if args.checkpoint is None else args.checkpoint
top_k = 5 if args.topk is None else args.topk


if arch == 'vgg16':
    model = models.vgg16(pretrained=True)
elif arch == 'vgg19':
    model = models.vgg19(pretrained=True)
elif arch == 'densenet121':
    model = models.densenet121(pretrained=True)
else:
    model = None

if model is None:
    print('Unknown model, please check your model name and try again')
    exit(0)

with open(category_names, 'r') as f:
    cat_to_name = json.load(f)

# num_features = model.classifier[-1].in_features
# # remove the last layer of model classifier
# features = list(model.classifier.children())[:-1]
# # set last layer's out features to the length of flower category
# features.extend([nn.Linear(num_features, len(cat_to_name))])
# model.classifier = nn.Sequential(*features)

map_location = 'cpu'
if args.gpu:
    if torch.cuda.is_available():
        map_location = 'cuda:0'

device = torch.device(map_location)

ckpt = torch.load(checkpoint, map_location=map_location)
trained_model = ckpt['model']
model.classifier = trained_model.classifier
# freeze layers of model
for param in model.parameters():
    param.require_grad = False

state_dict = ckpt['state_dict']
model.load_state_dict(state_dict)

class_to_idx = ckpt['class_to_idx']

idx_to_class = {v: k for k, v in class_to_idx.items()}

print(idx_to_class)

model = model.to(device)
#print(model)


def process_image(image):

    image_process = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    image_tensor = image_process(image)

    return image_tensor


def predict(image_path, model, topk):
    img = Image.open(image_path)
    img = process_image(img)

    # Convert 2D image to 1D vector
    img = np.expand_dims(img, 0)

    img = torch.from_numpy(img)

    model.eval()
    inputs = Variable(img).to(device)
    logits = model.forward(inputs)

    ps = F.softmax(logits, dim=1)
    topk = ps.cpu().topk(topk)

    return (e.data.numpy().squeeze().tolist() for e in topk)

# tested with cmd "python predict.py './flowers/test/1/image_06743.jpg' 'checkpoint_vgg16.pth'"
probs, classes = predict(args.input, model, top_k)
flower_names = [cat_to_name[idx_to_class[e]] for e in classes]

print(probs)
print(classes)
print(flower_names)