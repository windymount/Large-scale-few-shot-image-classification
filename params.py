"""

Hyper-parameters

"""
import torchvision.transforms as transforms


CIFAR100_TRANSFORM = transforms.Compose([    
    transforms.Resize(256),           
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )])
