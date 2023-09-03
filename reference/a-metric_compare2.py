import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma', 'DejaVu Sans','Lucida Grande', 'Verdana'][2]


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
res_conf = torch.load('results/metric_compre_conf.pt')

results = results[:,0,0,-1]
res_conf = res_conf[:,0,0]

print(results.shape)
print(res_conf.shape)

fig, ax = plt.subplots(1,4,figsize=(10,3))

size = [6,2,7,7]

for m_id, m in enumerate(metrics):
    a = size[m_id]
    ax[m_id].set_title('%s score = %0.3f' % (m, results[m_id]))
    
    ima = np.fliplr(np.flipud(res_conf[m_id][:a, :a])) if m_id == 1 else res_conf[m_id][:a, :a]
    
    ax[m_id].imshow(ima, cmap=cmaps[m_id])
    
    for i in range(ima.shape[0]):
        for j in range(ima.shape[1]):
            vala = ima[i, j]
            ax[m_id].text(j, i, '%.0f' % vala, 
                             ha='center', va='center', color='white' if vala > ima.mean() else colors[m_id], fontsize=8)
            
    ax[m_id].set_xticks([])
    ax[m_id].set_yticks([])


plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('figures/matrixes.png')
plt.savefig('figures/matrixes.eps')
