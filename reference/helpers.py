from torchvision import transforms
from torchosr.data import MNIST_base
from torchosr.models.architectures import fc_lower_stack

root = 'data'
t_mnist = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor()
])

def getdata(index):
    if index==0 or index==4:
        return MNIST_base(root=root, download=True, transform=t_mnist)
    print('Invalid argument')

def getls(index):
    if index==2:
        return fc_lower_stack(depth=1, img_size_x=28, n_out_channels=64)
    print('Invalid argument')

def get_openmax_epsilon(n_known):
    return {9:.1, 8:.1, 7:.1, 6:.1,
            5:.2, 4:.4, 3:.5, 2:.6}[n_known]
        
def get_softmax_epsilon(n_known):
    return {9:.05, 8:.1, 7:.2, 6:.25, 
            5:.35, 4:.48, 3:.6, 2:.8}[n_known]