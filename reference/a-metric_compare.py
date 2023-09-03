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

print(results_m.shape)

fig, ax = plt.subplots(2,1,figsize=(10,7), sharex=True, sharey=True)

for m_id, m in enumerate(metrics):
    ax[0].set_title('Softmax')
    ax[0].plot(results_m[m_id,0], label='WA '+ m, color=colors[m_id], alpha=0.9)
    if m_id in [0,1]:
        ax[0].plot(results_f1_m[m_id,0], label='F1 '+m, color=colors[m_id], alpha=0.9, ls='--')
    
    ax[1].set_title('Openmax')
    ax[1].plot(results_m[m_id,1], label='WA '+ m, color=colors[m_id], alpha=0.9)
    if m_id in [0,1]:
        ax[1].plot(results_f1_m[m_id,1], label='F1 '+m, color=colors[m_id], alpha=0.9, ls='--')

for aa in ax:
    aa.grid(ls=':')
    aa.spines['top'].set_visible(False)    
    aa.spines['right'].set_visible(False)
    aa.set_xlim(0,256)
    aa.set_ylabel('Metric value')
    aa.legend(frameon=False, ncol=4)

    
plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('figures/plots.png')

plt.clf()
res_conf = torch.load('results/metric_compre_conf.pt')
res_cof_m = torch.nanmean(res_conf, axis=1)

fig, ax = plt.subplots(2,4,figsize=(10,5))
ax[0, 0].set_ylabel('Softmax', fontsize=12)
ax[1, 0].set_ylabel('Openmax', fontsize=12)

size = [6,2,7,7]

for m_id, m in enumerate(metrics):
    a = size[m_id]
    ax[0, m_id].set_title(m)
    
    ima = np.fliplr(np.flipud(res_cof_m[m_id, 0][:a, :a])) if m_id == 1 else res_cof_m[m_id, 0][:a, :a]
    imb = np.fliplr(np.flipud(res_cof_m[m_id, 1][:a, :a])) if m_id == 1 else res_cof_m[m_id, 1][:a, :a]
    
    ax[0, m_id].imshow(ima, cmap=cmaps[m_id])
    ax[1, m_id].imshow(imb, cmap=cmaps[m_id])
    
    for i in range(ima.shape[0]):
        for j in range(ima.shape[1]):
            vala = ima[i, j]
            valb = imb[i, j]
            ax[0, m_id].text(j, i, '%.0f' % vala, 
                             ha='center', va='center', color='white' if vala > ima.mean() else colors[m_id], fontsize=8)
            
            ax[1, m_id].text(j, i, '%.0f' % valb, 
                             ha='center', va='center', color='white' if valb > imb.mean() else colors[m_id], fontsize=8)
        
        
    ax[0, m_id].set_xticks([])
    ax[0, m_id].set_yticks([])
    
    ax[1, m_id].set_xticks([])
    ax[1, m_id].set_yticks([])

plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('figures/matrixes.png')
