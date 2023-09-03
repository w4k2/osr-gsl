import numpy as np
import matplotlib.pyplot as plt
import torch

epochs = 256
n_methods = 2
n_splits = 5

kkc = [0,1,2,3,4,5]
uuc = [6,7]

metrics = ['Inner', 'Outer', 'Halfpoint', 'Overall']
colors = [plt.cm.Blues(np.linspace(0,1,4)[2]), 
          plt.cm.Greens(np.linspace(0,1,4)[2]), 
          plt.cm.Purples(np.linspace(0,1,4)[2]), 
          plt.cm.Reds(np.linspace(0,1,4)[2])]
cmaps = ['Blues', 'Greens', 'Purples', 'Reds']

results = torch.load('results/metric_compre.pt')
results_f1 = torch.load('results/metric_compre_f1.pt')
results_m = torch.nanmean(results, dim=1)
results_f1_m = torch.nanmean(results_f1, dim=1)
results_std = torch.std(results, dim=1)
results_f1_std = torch.std(results_f1, dim=1)
print(results_m.shape)

fig, ax = plt.subplots(2,1,figsize=(10,7), sharex=True, sharey=True)

for m_id, m in enumerate(metrics):
    ax[0].set_title('Softmax')
    ax[0].plot(results_m[m_id,0], label='WA '+ m, color=colors[m_id], alpha=0.9)
    ax[0].fill_between(np.arange(epochs), results_m[m_id,0]-results_std[m_id,0], results_m[m_id,0]+results_std[m_id,0], color=colors[m_id], alpha=0.2)
    
    if m_id in [0,1]:
        ax[0].fill_between(np.arange(epochs), results_f1_m[m_id,0]-results_f1_std[m_id,0], results_f1_m[m_id,0]+results_f1_std[m_id,0], color=colors[m_id], alpha=0.2)
        ax[0].plot(results_f1_m[m_id,0], label='F1 '+m, color=colors[m_id], alpha=0.9, ls='--')
    
    ax[1].set_title('Openmax')
    ax[1].plot(results_m[m_id,1], label='WA '+ m, color=colors[m_id], alpha=0.9)
    ax[1].fill_between(np.arange(epochs), results_m[m_id,1]-results_std[m_id,1], results_m[m_id,1]+results_std[m_id,1], color=colors[m_id], alpha=0.2)

    if m_id in [0,1]:
        ax[1].plot(results_f1_m[m_id,1], label='F1 '+m, color=colors[m_id], alpha=0.9, ls='--')
        ax[1].fill_between(np.arange(epochs), results_f1_m[m_id,1]-results_f1_std[m_id,1], results_f1_m[m_id,1]+results_f1_std[m_id,0], color=colors[m_id], alpha=0.2)

for aa in ax:
    aa.grid(ls=':')
    aa.spines['top'].set_visible(False)    
    aa.spines['right'].set_visible(False)
    aa.set_xlim(0,256)
    aa.set_ylabel('Metric value')
    aa.legend(frameon=False, ncol=4)

    
plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('figures/plots_std.png')
