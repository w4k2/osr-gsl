from torchvision import transforms
from torchosr.data import MNIST_base
from torchosr.data import configure_division, get_train_test
from torch.utils.data import DataLoader
from torchosr.architectures import fc_lower_stack
from models.GSL import GSL
import torch
from tqdm import tqdm

# Modelling parameters
root = 'data'
learning_rate = 1e-2
batch_size = 64
epochs = 128

# Evaluation parameters
repeats = 5         # openness repeats 
n_splits = 5        # classical validation splits
n_openness = 5      # openness values
q = 20              # number of quants in gaussian distribution sigma

sigmas = torch.linspace(0.1,5,q)

# Load dataset
t_mnist = transforms.Compose([transforms.Resize(28),
                              transforms.ToTensor()])
base = MNIST_base(root=root, 
                  download=True, 
                  transform=t_mnist)

# Prepare division
config, openness = configure_division(base, 
                                      repeats=repeats, 
                                      n_openness=n_openness,
                                      seed=1410)

# METRICS x CONFIG x FOLD x QUANTS x EPOCH
results = torch.full((4, len(config), n_splits, q, epochs), torch.nan)

pbar = tqdm(total=len(config)*n_splits*len(sigmas)*epochs)

# Iterating configurations
for config_idx, (kkc, uuc) in enumerate(config):
        # Iterate divisions
        for fold in range(n_splits):
            # print('- Fold %i' % fold)
            train_data, test_data = get_train_test(base, 
                                                   kkc, uuc,
                                                   root=root,
                                                   tunning=True,
                                                   fold=fold,
                                                   seed=1411,
                                                   n_folds=n_splits)
            
            train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
                        
            for sigma_id, sigma in enumerate(sigmas):
            
                # Initialize model
                model = GSL(n_known=len(kkc), lower_stack=fc_lower_stack(depth=1, img_size_x=28, n_out_channels=64), sigma=sigma, n_generated=0.5, normal=True) 
            
                # Initialize loss function
                loss_fn = torch.nn.CrossEntropyLoss()

                # Initialize optimizer
                optimizer_softmax = torch.optim.SGD(model.parameters(), lr=learning_rate)
                  
                for t in range(epochs):
                    # Train
                    model.train(train_data_loader, loss_fn, optimizer_softmax)
                    
                    # Test
                    scores = model.test(test_data_loader, loss_fn)
                    results[:, config_idx, fold, sigma_id, t] = torch.tensor(scores)
                    
                    pbar.update(1)                        
                        
                print(results[:,config_idx,fold])
                torch.save(results, 'results/e-gsl-sigma.pt')
pbar.close()