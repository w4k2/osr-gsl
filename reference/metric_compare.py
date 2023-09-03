from torchosr.data import get_train_test
from torch.utils.data import DataLoader
import torchosr as osr
import torch
from tqdm import tqdm
from helpers import getdata, getls, get_openmax_epsilon, get_softmax_epsilon

###################################################################################
# Order: Dataset, load_prev, tunning

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
    
    methods = [
    osr.models.Softmax(lower_stack=getls(ls_arg), n_known=len(kkc), epsilon=get_softmax_epsilon(len(kkc))),
    osr.models.Openmax(lower_stack=getls(ls_arg), n_known=len(kkc), epsilon=get_openmax_epsilon(len(kkc))),
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
            if t != epochs-1:
                (inner_score, outer_score, 
                 hp, overall) = model.test(test_data_loader, loss_fn)
                
                results[0, fold, model_id, t] = inner_score
                results[1, fold, model_id, t] = outer_score
                results[2, fold, model_id, t] = hp
                results[3, fold, model_id, t] = overall
            else:
                (inner_score, outer_score, hp, overall, 
                 c_inn, c_out, c_hp, c_over) = model.test(test_data_loader, loss_fn, conf=True)
                
                results[0, fold, model_id, t] = inner_score
                results[1, fold, model_id, t] = outer_score
                results[2, fold, model_id, t] = hp
                results[3, fold, model_id, t] = overall
                
                res_conf[0,fold,model_id, :c_inn.shape[0], :c_inn.shape[1]] = c_inn
                res_conf[1,fold,model_id, :c_out.shape[0], :c_out.shape[1]] = c_out
                res_conf[2,fold,model_id, :c_hp.shape[0], :c_hp.shape[1]] = c_hp
                res_conf[3,fold,model_id, :c_over.shape[0], :c_over.shape[1]] = c_over
                
            print(inner_score, outer_score, hp, overall)
            
            pbar.update(1)
                                        
        torch.save(results, 'results/metric_compre.pt')
        torch.save(res_conf, 'results/metric_compre_conf.pt')
    
pbar.close()