import numpy as np
import matplotlib.pyplot as plt

e1_om_sm = np.mean(np.load('results/e1_sm_om_imb_notune.npy').reshape(36,5,2), axis=1)
e1_om_sm_cc = np.mean(np.load('results/e1_sm_om_cc.npy').reshape(36,5,2), axis=1)
print(e1_om_sm.shape)
print(e1_om_sm_cc.shape)

ee = np.max(e1_om_sm)/800

fig, ax = plt.subplots(1,1,figsize=(7,7))
ax.set_title('Samples proportion in test set')
ax.scatter(10*e1_om_sm_cc[:,0]-1.5, 10*e1_om_sm_cc[:,1]-1.5,
           s=e1_om_sm[:,0]/ee, 
           c='black', label='Known samples', edgecolors='white', zorder=100, ls=':')
ax.scatter(10*e1_om_sm_cc[:,0]+1.5, 10*e1_om_sm_cc[:,1]+1.5, 
           s=e1_om_sm[:,1]/ee, 
           c='white', label='Unknown samples', edgecolors='black', zorder=100, ls=':')

ax.set_xlabel('# KKC')
ax.set_ylabel('# UUC')

grad = np.zeros((100,100))
x = np.linspace(0,10,100)
y = np.linspace(0,10,100)

for x_id, x_i in enumerate(x):
    for y_id, y_i in enumerate(y):
        tr = x_i
        te = x_i + y_i
        grad[x_id, y_id] = 1 - np.sqrt((2*tr)/(tr+te))
        
ax.imshow(grad, cmap='binary', origin='lower', interpolation='bilinear')
ax.set_xlim(15,95)
ax.set_ylim(5,85)

ax.set_xticks(np.linspace(20,90,8), np.arange(2,10))
ax.set_yticks(np.linspace(10,80,8), np.arange(1,9))

ax.grid(ls=':')

plt.legend()
plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('figures/owady_imb.png')

