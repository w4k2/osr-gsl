import numpy as np
import torch
from tabulate import tabulate

metrics = ['Inner', 'Outer', 'Halfpoint', 'Overall']

# Prepare division

from torchosr.data.base_datasets import MNIST_base
from torchosr.data import configure_division
root= 'data'
base = MNIST_base(root=root, download=True)
config, openness = configure_division(base, 
                                      repeats=5, 
                                      n_openness=5,
                                      seed=19763)

order = np.argsort(openness)
print(openness)
print(openness[order])


for m_id, m in enumerate(metrics):
        
    rows = []
    rows.append(['Dataset', 'Openness', 'SM', 'OM', 'GSL'])

    #Part1 -- MNIST FC
    repeats = 5
    n_splits = 5
    n_openness = 5

    res = torch.load('results/e-compare_mnist_fc.pt') #metrics, configs, splits, methods, epochs
    res = res[m_id] # configs, splits, methods, epochs

    res = res.reshape(n_openness, repeats * n_splits, 3, -1)
    # Just last epoch
    res = res[:,:,:,-1]

    res_mean = torch.mean(res, dim=1)
    res_sigm = torch.std(res, dim=1)
    
    
    for ord_id, ord in enumerate(order):
        
        rows.append([
            'MNIST', 
            'O=%.3f' % openness[ord], 
            '%.3f(%.3f)' % (res_mean[ord, 0], res_sigm[ord, 0]),
            '%.3f(%.3f)' % (res_mean[ord, 1], res_sigm[ord, 1]),
            '%.3f(%.3f)' % (res_mean[ord, 2], res_sigm[ord, 2]),
        ])
    
    
    # Part2 -- CIFAR10

    res = torch.load('results/e-compare_c10_osrci.pt') #metrics, configs, splits, methods, epochs
    res = res[m_id] # configs, splits, methods, epochs

    res = res.reshape(n_openness, repeats * n_splits, 3, -1)
    # Just last epoch
    res = res[:,:,:,-1]

    res_mean = torch.mean(res, dim=1)
    res_sigm = torch.std(res, dim=1)
    
    for ord_id, ord in enumerate(order):

        rows.append([
            'CIFAR10', 
            'O=%.3f' % openness[ord], 
            '%.3f(%.3f)' % (res_mean[ord, 0], res_sigm[ord, 0]),
            '%.3f(%.3f)' % (res_mean[ord, 1], res_sigm[ord, 1]),
            '%.3f(%.3f)' % (res_mean[ord, 2], res_sigm[ord, 2]),
        ])
        
    print(tabulate(rows))
    with open('tab/%s.txt' % m, 'w') as f:
        f.write(tabulate(rows, tablefmt='latex'))
    # exit()