import torch
import matplotlib.pyplot as plt
import numpy as np
from torchosr.data.base_datasets import MNIST_base
from torchosr.data import configure_division

# Get openenss

repeats = 5
n_splits = 5
n_openness = 5

# Load dataset
base = MNIST_base(root='data', download=True)
config, openness = configure_division(base, repeats=repeats, n_openness=n_openness, seed=1410)
order = np.argsort(openness)
print(openness[order])

q = 20
ep = 128

sigmas = torch.linspace(0.1,5,q)

res_sig = torch.load('results/e-gsl-sigma.pt')
print(res_sig.shape)
res_sig = res_sig.reshape(4,5,25,q,ep)
res_sig_m = torch.nanmean(res_sig, 2)
print(res_sig.shape)
print(res_sig_m.shape)

fig, ax = plt.subplots(5,5,figsize=(12,8), sharex=True, sharey=True)

for m_id, m in enumerate(['Inner', 'Outer', 'Halfpoint', 'Overall', 'Combined']):
    for i in range(n_openness):
        
        if m_id == 0:
            ax[i,m_id].set_ylabel('Op=%.3f \n sigma'  % openness[order[i]].item())
        if i == 0:
            ax[i,m_id].set_title(m)
            
        if m_id == 4:
            #Combine
            img = 1-res_sig_m[1:, order[i]].swapaxes(0,-1).swapaxes(0,1)
            ax[i,m_id].imshow(img, aspect='auto')

        else:       
            ax[i,m_id].imshow(res_sig_m[m_id, order[i]], cmap='coolwarm', vmin=0, vmax=1,  aspect='auto')
            best_id = np.argmax(res_sig_m[m_id, order[i],:,-1])
            print(m, '\n', res_sig_m[m_id, order[i]])
            ax[i,m_id].hlines(best_id,0,ep,ls=':',color='r')


        ax[i,m_id].set_yticks(np.arange(len(sigmas))[::3], ['%.1f' % s.item() for s in sigmas][::3])        
        ax[i,m_id].set_xlim(0,ep)
        

plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('fig/gsl_sigma.png')
