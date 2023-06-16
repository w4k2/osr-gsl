from torchosr.data import configure_division, get_train_test
from torch.utils.data import DataLoader
import torchosr as osr
import torch
from tqdm import tqdm
from torchvision import transforms
from torchosr.data import CIFAR10_base
from torchosr.architectures import osrci_lower_stack
from torchosr.utils import get_softmax_epsilon, get_openmax_epsilon
from models.GSL import GSL
import time

def getls():
    return osrci_lower_stack(n_out_channels=64, depth=3, img_size_x=32)

# Modelling parameters
learning_rate = [1e-3, 1e-3, 1e-2]
batch_size = 64
epochs = 128

# Evaluation parameters
repeats = 1         # openness repeats
n_splits = 5        # classical validation splits
n_openness = 5      # openness values

# Load dataset
root= 'data'

t_cifar = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()])

base = CIFAR10_base(root=root, download=True, transform=t_cifar)

# Prepare division
config, openness = configure_division(base,
                                    repeats=repeats, 
                                    n_openness=n_openness,
                                    seed=1410)

n_methods = 3
n_metrics = 4

interval=10

results = torch.full((len(config), n_splits, n_methods, 2), torch.nan)
pbar = tqdm(total=len(config)*n_splits*n_methods)

# Iterating configurations
for config_idx, (kkc, uuc) in enumerate(config):
        print('# Configuration %i [openness %.3f]' % (config_idx, openness[config_idx // repeats]), 
              'known:', kkc.numpy(), 
              'unknown:', uuc.numpy())
        # Iterate divisions
        for fold in range(n_splits):
            train_data, test_data = get_train_test(base, 
                                                   kkc, uuc,
                                                   root='data',
                                                   tunning=False,
                                                   fold=fold,
                                                   seed=1411,
                                                   n_folds=n_splits)
            
            train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
            
            methods = [
            osr.models.TSoftmax(lower_stack=getls(), n_known=len(kkc), epsilon=get_softmax_epsilon(len(kkc))),
            osr.models.Openmax(lower_stack=getls(), n_known=len(kkc), epsilon=get_openmax_epsilon(len(kkc))),
            GSL(lower_stack=getls(), n_known=len(kkc), sigma=1.2, n_generated=.5, normal=True),
            ]
            
            for model_id, model in enumerate(methods):                         
                # Initialize loss function
                loss_fn = torch.nn.CrossEntropyLoss()

                # Initialize optimizer
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate[model_id])

                s = time.time()
                model.train(train_data_loader, loss_fn, optimizer)
                tr_time = time.time()-s
                results[config_idx, fold, model_id, 0] = tr_time
                
                s = time.time()
                model.test(test_data_loader, loss_fn)
                te_time = time.time()-s
                results[config_idx, fold, model_id, 1] = te_time
                    
                pbar.update(1)
                        
                print(config_idx, fold, model_id, results[config_idx, fold, model_id])                    
                torch.save(results, 'results/e-time_c10.pt')
pbar.close()