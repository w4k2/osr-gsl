from torchosr.data import get_train_test
from torch.utils.data import DataLoader
import torchosr as osr
import torch
from tqdm import tqdm
from helpers import getdata, getls, get_openmax_epsilon, get_softmax_epsilon
from torchmetrics import ConfusionMatrix, F1Score
from torch import nn
from scipy.stats import exponweib
import numpy as np


###################################################################################
# Order: Dataset, load_prev, tunning

def omtest(self, dataloader, loss_fn): 
    inner_preds = []
    inner_preds_harry_potter = []
    inner_preds_overall = []
    outer_preds = []
    
    inner_targets = []
    outer_targets = []
    overall_targets = []
    
    # Define metric for open set detection
    try:
        inner_metric = F1Score(task="multiclass", num_classes=self.n_known, average='weighted')
        overall_metric = F1Score(task="multiclass", num_classes=self.n_known+1, average='weighted')
    except:
        inner_metric = None
        overall_metric = None
    outer_metric = F1Score(task='binary', average='weighted')

    with torch.no_grad():
        for X, y in dataloader:
            # Get y flat and known mask
            y_flat = y.argmax(1)
            known_mask = ~(y_flat == self.n_known)
            
            # Calculate logits for full batch
            logits = self(X)
            slogit = logits.sum(1)
            
            # Rank activations for alpha trimming
            a_trimmer = 1 - torch.nn.functional.one_hot(torch.topk(logits, self.alpha, dim=1)[1],
                                                        logits.shape[1]).sum(1)
                            
            # Calculate weights
            w = np.ones(logits.shape).astype(float)
            for i in self.weibs:
                rv = self.weibs[i]
                w[:,i] -= exponweib.cdf(logits[:,i], *rv)
            
            w = torch.Tensor(w)
            try:
                w[a_trimmer] = 1
            except:
                print('[a_trimmer] Skipped', w, a_trimmer)
            
            # Establish weighted logits and unknown activation
            v_logits = w*logits
            plogit = v_logits.sum(1)
            unknown_activation = torch.sub(slogit, plogit)
            # print('aa', unknown_activation)

            # Calculate corrected activation
            logits = torch.cat((v_logits, unknown_activation[:,None]), 1)          

            # Establish inner preds and inner y
            inner_pp = nn.Softmax(dim=1)(logits[:,:-1])
            inner_pred = inner_pp.argmax(1)[known_mask]
            inner_target = y_flat[known_mask]           
                            
            # Establish outer pred
            # Here is softmax
            #outer_pred = (nn.Softmax(dim=1)(logits).max(1).values > self.epsilon).int()
            # Czy wsparcie jest dla klasy znanej
            pred_known = (nn.Softmax(dim=1)(logits).argmax(1) != self.n_known).int() # 1=KKC

            # Czy wsparcie jest dostateczne do zachowania decyzji
            pred_sure = (nn.Softmax(dim=1)(logits).max(1).values > self.epsilon).int() # jeżeli wsparcie jest większe to KKC

            # pred_known = 1-pred_known
            outer_pred = pred_known * pred_sure
            outer_target = (y_flat != self.n_known).int()
            
            inner_pp_overall = nn.Softmax(dim=1)(logits)
            inner_pred_overall = inner_pp_overall.argmax(1)
            inner_pred_overall[outer_pred==0]=self.n_known
            inner_pred_harry_potter = inner_pred_overall[known_mask]
            
            # Store predictions
            inner_preds.append(inner_pred)
            inner_preds_harry_potter.append(inner_pred_harry_potter)
            inner_preds_overall.append(inner_pred_overall)
            inner_targets.append(inner_target)
            overall_targets.append(y_flat)
            
            outer_preds.append(outer_pred)
            outer_targets.append(outer_target)
            
    inner_targets = torch.cat(inner_targets)
    overall_targets = torch.cat(overall_targets)
    inner_preds = torch.cat(inner_preds)
    inner_preds_harry_potter = torch.cat(inner_preds_harry_potter)
    inner_preds_overall = torch.cat(inner_preds_overall)
    
    outer_targets = torch.cat(outer_targets)
    outer_preds = torch.cat(outer_preds)
    
    # Calculate scores
    if inner_metric is not None:
        inner_score = inner_metric(inner_preds, inner_targets)
        inner_score_harry = overall_metric(inner_preds_harry_potter, inner_targets)
        inner_score_overall = overall_metric(inner_preds_overall, overall_targets)
    else:
        inner_score = torch.nan
        inner_score_harry = torch.nan
        inner_score_overall = torch.nan
    print('open',torch.unique(outer_preds, return_counts=True))
    print(ConfusionMatrix(task="multiclass", num_classes=self.n_known+1)(inner_preds_harry_potter, inner_targets))
    print(ConfusionMatrix(task="multiclass", num_classes=self.n_known+1)(inner_preds_overall, overall_targets))
    outer_score = outer_metric(outer_preds, outer_targets)
    print(outer_score)
    
    if self.verbose:
        print('Inner metric : %.3f' % inner_score)
        print('Outer metric : %.3f' % outer_score)
        
    return inner_score, outer_score, inner_score_harry, inner_score_overall

def smtest(self, dataloader, loss_fn):
    inner_preds = []
    inner_preds_harry_potter = []
    inner_preds_overall = []
    outer_preds = []
    
    inner_targets = []
    outer_targets = []
    overall_targets = []
    
    # Define metric for open set detection
    try:
        inner_metric = F1Score(task="multiclass", num_classes=self.n_known, average='weighted')
        overall_metric = F1Score(task="multiclass", num_classes=self.n_known+1, average='weighted')
    except:
        inner_metric = None
    outer_metric = F1Score(task='binary', average='weighted')

    with torch.no_grad():
        for X, y in dataloader:
            # Get y flat and known mask
            y_flat = y.argmax(1)
            known_mask = ~(y_flat == self.n_known)
            
            # Calculate logits for full batch
            logits = self(X)

            # Establish inner preds and inner y
            inner_pp = nn.Softmax(dim=1)(logits)
            overall_pred = inner_pp.argmax(1)
            inner_pred = inner_pp.argmax(1)[known_mask]
            inner_target = y_flat[known_mask]
            
            # Establish outer pred
            # Here is softmax
            outer_pred = (nn.Softmax(dim=1)(logits).max(1).values > self.epsilon).int()
            outer_target = (y_flat != self.n_known).int()
                            
            overall_pred[outer_pred==0]=self.n_known
            inner_pred_harry_potter = overall_pred[known_mask]
            
            # Store predictions
            inner_preds.append(inner_pred)
            inner_preds_harry_potter.append(inner_pred_harry_potter)
            inner_preds_overall.append(overall_pred)

            inner_targets.append(inner_target)
            overall_targets.append(y_flat)
            
            outer_preds.append(outer_pred)
            outer_targets.append(outer_target)
            
    inner_targets = torch.cat(inner_targets)
    overall_targets = torch.cat(overall_targets)
    inner_preds = torch.cat(inner_preds)
    inner_preds_harry_potter = torch.cat(inner_preds_harry_potter)
    inner_preds_overall = torch.cat(inner_preds_overall)
    
    outer_targets = torch.cat(outer_targets)
    outer_preds = torch.cat(outer_preds)
    
    # Calculate scores
    if inner_metric is not None:
        inner_score = inner_metric(inner_preds, inner_targets)
        inner_score_harry = overall_metric(inner_preds_harry_potter, inner_targets)
        inner_score_overall = overall_metric(inner_preds_overall, overall_targets)
    else:
        inner_score = torch.nan
        inner_score_harry = torch.nan
        inner_score_overall = torch.nan
        
    outer_score = outer_metric(outer_preds, outer_targets)
    
    print(ConfusionMatrix(task="multiclass", num_classes=self.n_known+1)(inner_preds_harry_potter, inner_targets))
    print(ConfusionMatrix(task="multiclass", num_classes=self.n_known+1)(inner_preds_overall, overall_targets))
    
    if self.verbose:   
        print('Inner metric : %.3f' % inner_score)
        print('Outer metric : %.3f' % outer_score)
        
    return inner_score, outer_score, inner_score_harry, inner_score_overall

# Datasets:
#  4 - MNIST + FC
dataset_arg = 4 #mnist
ls_arg = 2 #fc

###################################################################################

# Modelling parameters
learning_rate = 1e-4
batch_size = 64
epochs = 256

# Load dataset
base = getdata(dataset_arg)

n_methods = 2
n_splits = 5

kkc = [0,1,2,3,4,5]
uuc = [6,7]

results = torch.full((4, n_splits, n_methods, epochs), torch.nan)
res_conf = torch.full((4, n_splits, n_methods, len(kkc)+1, len(kkc)+1), torch.nan)

pbar = tqdm(total=n_methods*epochs*n_splits)

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
    
    SM = osr.models.Softmax
    OM = osr.models.Openmax
    
    SM.test = smtest
    OM.test = omtest
    
    methods = [
        SM(lower_stack=getls(ls_arg), n_known=len(kkc), epsilon=get_softmax_epsilon(len(kkc))),
        OM(lower_stack=getls(ls_arg), n_known=len(kkc), epsilon=get_openmax_epsilon(len(kkc))),
    ]
    
    for model_id, model in enumerate(methods):                         
        # Initialize loss function
        loss_fn = torch.nn.CrossEntropyLoss()

        # Initialize optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
        for t in range(epochs):
            # Train
            model.train(train_data_loader, loss_fn, optimizer)
            
            # Test
            inner_score, outer_score, hp, overall = model.test(test_data_loader, loss_fn)
            results[0, fold, model_id, t] = inner_score
            results[1, fold, model_id, t] = outer_score
            results[2, fold, model_id, t] = hp
            results[3, fold, model_id, t] = overall
            
            print(inner_score, outer_score, hp, overall)
            
            pbar.update(1)
                                        
        torch.save(results, 'results/metric_compre_f1.pt')
    
pbar.close()