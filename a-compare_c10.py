import torch
import matplotlib.pyplot as plt
import numpy as np
from torchosr.data.base_datasets import MNIST_base
from torchosr.data import configure_division

root= 'data'
base = MNIST_base(root=root, download=True)

green = 'forestgreen'
orange = '#ED722E'
blue = '#1E4CF5'

# Prepare division
config, openness = configure_division(base, 
                                      repeats=5, 
                                      n_openness=5,
                                      seed=19763)

order = np.argsort(openness)
print(openness)
print(openness[order])

####

res = torch.load('results/e-compare_c10_osrci.pt')
res = res.reshape(4,5,25,3,128)
res_m = torch.nanmean(res, 2)
print(res.shape)
print(res_m.shape)

cols = ['blue', 'purple', 'red']
cols = [green, blue, orange]
fig, ax = plt.subplots(5, 4, figsize=(7,7), sharex=True, sharey=True)

for i in range(5):
    for m_id, m in enumerate(['Inner', 'Outer', 'Halfpoint', 'Overall']):
        
        if order[i]==0:
            ax[order[i], m_id].set_title(m)
            ax[-1, m_id].set_xlabel('epoch')
        if m_id==0:
            ax[order[i], m_id].set_ylabel('$O=%.3f$\nWeighted Acc.'% openness[order[i]], fontsize=10)
        
        for method_id, method in enumerate(['SM', 'OM', 'GSL']):
            ax[order[i],m_id].plot(res_m[m_id, order[i], method_id], 
                                   c=cols[method_id], 
                                   label=method, alpha=0.6)

ax[0,0].legend(ncols=1, frameon=False)

for a in ax.ravel():
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)
    a.set_xlim(0, 128)
    a.set_yticks(np.linspace(0,1,5))
    a.grid(ls=":")

plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('fig/compare-c10.png')
plt.savefig('fig/compare-c10.eps')