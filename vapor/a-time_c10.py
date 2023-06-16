import torch
import matplotlib.pyplot as plt
import numpy as np

results = torch.load('results/e-time_c10.pt') # config x folds x methods x (training, testing)
print(results.shape)
results_mean = np.mean(results.detach().numpy(), axis=(0,1))
print(results_mean)

fig, ax = plt.subplots(1,1,figsize=(7,5))

ax.bar(np.arange(3)-0.2, results_mean[:,0], label='training', width=0.35)
ax.bar(np.arange(3)+0.2, results_mean[:,1], label='testing', width=0.35)
ax.set_xticks(np.arange(3), ['SM', 'OM', 'GSL'])
ax.legend()
plt.tight_layout()
plt.savefig('foo.png')