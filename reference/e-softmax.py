from torchvision import transforms
from torchosr.data import MNIST_base
from torchosr.data import configure_division, get_train_test
from torch.utils.data import DataLoader
from torchosr.models import fc_lower_stack
from torchmetrics import Accuracy
import torchosr as osr
import torch
from torch import nn
from tqdm import tqdm

# Modelling parameters
root = 'data'
learning_rate = 1e-3
batch_size = 64
epochs = 64

# Evaluation parameters
repeats = 5         # openness repeats 
n_splits = 5        # classical validation splits
q = 20             # number of epsilon quants

epsilons = torch.linspace(0,1,q)

# Load dataset
t_mnist = transforms.Compose([transforms.Resize(28),
                              transforms.ToTensor()])
base = MNIST_base(root=root, 
                  download=True, 
                  transform=t_mnist)

# Prepare division
config, openness = configure_division(base, 
                                      repeats=repeats, 
                                      seed=1410)

# CONFIG x FOLD x EPSILON-QUANT x EPOCH
results = torch.zeros((4, len(config), n_splits, q, epochs))

# Iterating configurations
for config_idx, (kkc, uuc) in enumerate(tqdm(config)):
        # Estimate and store openness of a config
        openness_idx = config_idx // repeats
        print('# Configuration %i [openness %.3f]' % (config_idx,
                                                      openness[openness_idx]), 
              'known:', kkc.numpy(), 
              'unknown:', uuc.numpy())
        
        # Iterate divisions
        for fold in range(n_splits):
            train_data, test_data = get_train_test(base, 
                                                   kkc, uuc, 
                                                   root=root, 
                                                   tunning=True, 
                                                   fold=fold, 
                                                   seed=1411,
                                                   n_folds=n_splits)
            
            train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)            
                    
            # Lower stack
            ls = fc_lower_stack(depth=1,
                                img_size_x=28, 
                                n_out_channels=64)
            
            # Initialize model
            model = osr.models.Softmax(n_known=len(kkc),
                                       lower_stack=ls)
        
            # Initialize loss function
            loss_fn = torch.nn.CrossEntropyLoss()

            # Initialize optimizer
            optimizer_softmax = torch.optim.SGD(model.parameters(), lr=learning_rate)
            
            # Define metric for open set detection
            inner_metric = Accuracy(task="multiclass", num_classes=model.n_known, average='weighted')    
            overall_metric = Accuracy(task="multiclass", num_classes=model.n_known+1, average='weighted')    
            outer_metric = Accuracy(task='binary', average='weighted')
    
            for t in range(epochs):                                
                # Train
                model.train(train_data_loader, loss_fn, optimizer_softmax)
        
                # Test
                inner_preds = [[] for _ in range(q)]
                inner_preds_harry_potter = [[] for _ in range(q)]
                inner_preds_overall = [[] for _ in range(q)]
                outer_preds = [[] for _ in range(q)]
                
                inner_targets = [[] for _ in range(q)]
                outer_targets = [[] for _ in range(q)]
                overall_targets = [[] for _ in range(q)]

                with torch.no_grad():
                    for X, y in test_data_loader:
                        # Get y flat and known mask
                        y_flat = y.argmax(1)
                        known_mask = ~(y_flat == model.n_known)                        
                        
                        # Calculate logits for full batch
                        logits = model(X)
                                                                        
                        # Establish outer pred
                        # Here is softmax
                        for e_idx, epsilon in enumerate(epsilons):
                            
                            # Establish inner preds and inner y
                            inner_pp = nn.Softmax(dim=1)(logits)
                            overall_pred = inner_pp.argmax(1)
                            inner_pred = overall_pred[known_mask]
                            inner_target = y_flat[known_mask]
                            
                            # Establish outer pred
                            # Here is softmax
                            outer_pred = (nn.Softmax(dim=1)(logits).max(1).values > epsilon).int()
                            outer_target = (y_flat != model.n_known).int()
                                                                        
                            overall_pred[outer_pred==0] = model.n_known
                            inner_pred_harry_potter = overall_pred[known_mask]
                                        
                            inner_preds[e_idx].append(inner_pred)
                            inner_preds_harry_potter[e_idx].append(inner_pred_harry_potter)
                            inner_preds_overall[e_idx].append(overall_pred)
                            outer_preds[e_idx].append(outer_pred)
                            
                            inner_targets[e_idx].append(inner_target)
                            outer_targets[e_idx].append(outer_target)
                            overall_targets[e_idx].append(y_flat)
                                    
                for e_idx, epsilon in enumerate(epsilons):
                    
                    _inner_targets = torch.cat(inner_targets[e_idx])
                    _overall_targets = torch.cat(overall_targets[e_idx])
                    _inner_preds = torch.cat(inner_preds[e_idx])
                    _inner_preds_harry_potter = torch.cat(inner_preds_harry_potter[e_idx])
                    _inner_preds_overall = torch.cat(inner_preds_overall[e_idx])
                         
                    _outer_targets = torch.cat(outer_targets[e_idx])
                    _outer_preds = torch.cat(outer_preds[e_idx])
                    
                    inner_score = inner_metric(_inner_preds, _inner_targets)
                    inner_score_harry = overall_metric(_inner_preds_harry_potter, _inner_targets)
                    inner_score_overall = overall_metric(_inner_preds_overall, _overall_targets)
                    outer_score = outer_metric(_outer_preds, _outer_targets)
                    
                    results[0, config_idx, fold, e_idx, t] = inner_score
                    results[1, config_idx, fold, e_idx, t] = outer_score
                    results[2, config_idx, fold, e_idx, t] = inner_score_harry
                    results[3, config_idx, fold, e_idx, t] = inner_score_overall
                    
                    # print(inner_score, outer_score, inner_score_harry, inner_score_overall)
                    
            torch.save(results, 'results/e-softmax2.pt')
