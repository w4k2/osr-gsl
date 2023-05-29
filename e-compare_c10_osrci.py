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

torch.set_num_threads(4)

def getls():
    return osrci_lower_stack(n_out_channels=64, depth=3, img_size_x=32)

# Modelling parameters
learning_rate = [1e-3, 1e-3, 1e-2]
batch_size = 64
epochs = 128

# Evaluation parameters
repeats = 5         # openness repeats
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

results = torch.full((n_metrics, len(config), n_splits, n_methods,epochs), torch.nan)

pbar = tqdm(total=len(config)*n_splits*n_methods*epochs)

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
            GSL(lower_stack=getls(), n_known=len(kkc), sigma=1.2, n_generated=1./len(kkc), normal=True),
            ]
            
            for model_id, model in enumerate(methods):                         
                # Initialize loss function
                loss_fn = torch.nn.CrossEntropyLoss()

                # Initialize optimizer
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate[model_id])

                for t in range(epochs):
                    # Train
                    model.train(train_data_loader, loss_fn, optimizer)
                    
                    # Test
                    scores = model.test(test_data_loader, loss_fn)
                    results[:, config_idx, fold, model_id, t] = torch.tensor(scores)
                    
                    pbar.update(1)
                    
                    print(config_idx, fold, model_id, results[:,config_idx, fold, model_id, t])                    
                torch.save(results, 'results/e-compare_c10_osrci.pt')
pbar.close()