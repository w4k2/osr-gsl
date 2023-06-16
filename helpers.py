from torchvision import transforms
from torchosr.data import MNIST_base, SVHN_base, CIFAR10_base, CIFAR100_base
from torchosr.models.architectures import osrci_lower_stack, fc_lower_stack
from torch import nn
from torchosr.utils.base import inverse_transform


root='data'
t_mnist = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor()])

t_svhn = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()])

t_cifar = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()])

t_omni = transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            inverse_transform()])

def getdata(index):
    if index==0 or index==4:
        return MNIST_base(root=root, download=True, transform=t_mnist)
    if index==1:
        return SVHN_base(root=root, download=True, transform=t_svhn)
    if index==2:
        return CIFAR10_base(root=root, download=True, transform=t_cifar)
    if index==3:
        return CIFAR100_base(root=root, download=True, transform=t_cifar)
    print('Invalid argument')
       

def getls(index):
    if index==0:
        return osrci_lower_stack(depth=1, img_size_x=28, n_out_channels=64)
    if index==1:
        return alexNet32_lower_stack2(n_out_channels=64)
    if index==2:
        return fc_lower_stack(depth=1, img_size_x=28, n_out_channels=64)
    if index==3:
        return osrci_lower_stack2(depth=3, img_size_x=28, n_out_channels=64)
    print('Invalid argument')
    
def  osrci_lower_stack2(depth, img_size_x=28, n_out_channels=64):
    return nn.Sequential(
        nn.Conv2d(depth, 10, kernel_size=5),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Conv2d(10, 20, kernel_size=5),
        nn.Dropout2d(),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(500, n_out_channels),
        nn.ReLU(),
        )
    
    
def alexNet32_lower_stack2(n_out_channels): 
    """
    Returns modified architecture based on AlexNet, suitable for size (32 x 32 x 3) images.

    :type n_out_channels: int
    :param n_out_channels: Size output

    :rtype: torch.nn.Sequential
    :returns: Lower stack sequential architecture
    """
    return nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3), #30 x 30 x 96
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 15 x 15 x 96
            
            nn.Conv2d(96, 256, kernel_size=3), # 13 x 13 x 256
            nn.ReLU(),
            # # nn.MaxPool2d(kernel_size=2, stride=2), # 8 x 8 x 256
                        
            # nn.Conv2d(256, 384, kernel_size=3, padding=1), # 13 x 13 x 384
            # nn.ReLU(),
            
            # nn.Conv2d(384, 384, kernel_size=3, padding=1), # 13 x 13 x 384
            # nn.ReLU(),
            
            # nn.Conv2d(384, 256, kernel_size=3, padding=1), # 13 x 13 x 256
            # nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=3, stride=2), # 6 x 6 x 256
            
            # nn.Dropout(),

            nn.Flatten(),
            
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            
            # nn.Dropout(),
            
            nn.Linear(4096, n_out_channels),
            nn.ReLU(),
            )
    

def get_openmax_epsilon(n_known):
    if n_known>5:
        return 0
    if n_known==5:
        return 0.18
    if n_known==4:
        return 0.28
    if n_known==3:
        return 0.4
    if n_known==2:
        return 0.5
        
def get_softmax_epsilon(n_known):
    if n_known==9:
        return 0.15
    if n_known==8:
        return 0.2
    if n_known==7:
        return 0.28
    if n_known==6:
        return 0.32
    if n_known==5:
        return 0.39
    if n_known==4:
        return 0.48
    if n_known==3:
        return 0.6
    if n_known==2:
        return 0.78