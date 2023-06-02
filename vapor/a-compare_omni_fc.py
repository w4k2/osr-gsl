import torch
import matplotlib.pyplot as plt
import numpy as np
from torchosr.data.base_datasets import MNIST_base, Omniglot_base
from torchosr.data import configure_division_outlier

root= 'data'
base = MNIST_base(root=root, download=True)
out = Omniglot_base(root=root, download=True)

# Prepare division
config, openness = configure_division_outlier(base,
                                            out,
                                            repeats=5, 
                                            n_openness=5,
                                            seed=1410)

order = np.argsort(openness)
print(openness)
print(openness[order])

####

interval=15

res = torch.load('results/e-compare_omni_fc.pt')
res = res.reshape(4,5,3,3,-1)
res_m = torch.nanmean(res, 2)
print(res.shape)
print(res_m.shape)

cols = ['blue', 'purple', 'red']
fig, ax = plt.subplots(5, 4, figsize=(14,10), sharex=True, sharey=True)

for i in range(5):
    for m_id, m in enumerate(['Inner', 'Outer', 'Halfpoint', 'Overall']):
        
        if order[i]==0:
            ax[order[i], m_id].set_title(m)
            ax[-1, m_id].set_xlabel('epoch')
        if m_id==0:
            ax[order[i], m_id].set_ylabel('WA \n Openness=%.3f'% openness[order[i]])
        
        for method_id, method in enumerate(['SM', 'OM', 'GSL']):
            ax[order[i],m_id].plot(np.arange(0,res.shape[-1]*interval,interval), res_m[m_id, order[i], method_id], c=cols[method_id], label=method, alpha=0.6)

        ax[order[i],m_id].grid(ls=':')

ax[0,0].legend(ncols=3, frameon=False)

plt.tight_layout()
plt.savefig('foo.png')