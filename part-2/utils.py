###########
# Imports #
###########
# Core
import os
import json

# 3rd-party
import torch
import torchvision.models as models
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



def base_model(arch):
    arch = arch.lower()
    
    if arch == 'vgg11':
        model = models.vgg11(pretrained=True)
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif arch == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif arch == 'resnet34':
        model = models.resnet34(pretrained=True)
    elif arch == 'squeezenet1_0':
        model = models.squeezenet1_0(pretrained=True)
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif arch == 'densenet169':
        model = models.densenet169(pretrained=True)
    elif arch == 'inception_v3':
        model = models.inception_v3(pretrained=True)
    else:
        print('Did not recognize model architecture. Using VGG13')
        model = models.vgg13(pretrained=True)
        
    # turn-off gradients
    for param in model.parameters():
        param.requires_grad = False
        
    return model


def get_valid_device(gpu):
    if gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    return device


def cat_to_name(cat, filename):
    """
    Maps category number to human readable name
    """
    
    with open(filename, 'r') as f:
        cat_to_name = json.load(f)
    
    return cat_to_name[str(cat)]


def save_model(filepath, model, epochs, optimizer, class_to_idx):
    checkpoint = {
        # 'input_size': node_sizes[0],
        # 'output_size': node_sizes[-1],
        #'features': model.features,
        'idx_to_class' : {str(v): str(k) for k,v in class_to_idx.items()},
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'epochs': epochs,
        # disabled because of storage constraints
        # 'optimizer_state_dict': optimizer.state_dict
    }

    torch.save(checkpoint, filepath)
    
    print(f'Saved model checkpoint to: {filepath}')

    
def load_model(filepath):
    def get_model_arch(filename):
        """
            Extracts the architecture handle from a checkpoint filename

            i.e. vgg13-accuracy-85.53.pth
        """
        return filename.split('-')[0]
    
    arch = get_model_arch(os.path.basename(filepath))
    model = base_model(arch)
    
    checkpoint = torch.load(filepath)
    # checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
    # model.features = checkpoint['features']
    model.idx_to_class = checkpoint['idx_to_class']
    model.classifier = checkpoint['classifier']
    model.state_dict = checkpoint['state_dict']
    return model


def process_image(image_path):
    ''' 
        Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    def image_transform(image_path):
        valid_transforms = transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225]
            )
        ])

        image = Image.open(image_path)
        image = valid_transforms(image).float()
        return np.array(image)
    
    return image_transform(image_path)



def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax