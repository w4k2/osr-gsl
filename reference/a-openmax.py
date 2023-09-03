import torch
from torchosr.data import configure_division
from torchosr.data import MNIST_base
from torchvision import transforms
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import numpy as np

# CONFIG x FOLD x EPSILON-QUANT x EPOCH
results = torch.load('results/e-openmax2.pt')

root = 'data'

# Evaluation parameters
repeats = 5         # openness repeats 
n_splits = 5       # classical validation splits
q = 20
epochs = 64
epsilons = torch.linspace(0,1,q)

t_mnist = transforms.Compose([transforms.Resize(28),
                              transforms.ToTensor()])
base = MNIST_base(root=root, 
                  download=True, 
                  transform=t_mnist)

# Prepare division
config, openness = configure_division(base, 
                                      repeats=repeats, 
                                      seed=1410)

# cmaps = ['cividis', 'coolwarm', 'magma', 'jet']
cmaps = ['Blues', 'Greens', 'Purples', 'Reds']
cmaps_ = [plt.cm.Blues(np.linspace(0,1,2)[1]), plt.cm.Greens(np.linspace(0,1,2)[1]), plt.cm.Purples(np.linspace(0,1,2)[1]), plt.cm.Reds(np.linspace(0,1,2)[1])]
for metric_idx, metric in enumerate(['inner', 'outer', 'HP', 'overall']):
    
    fig, ax = plt.subplots(8,8,figsize=(12,12), sharex=True, sharey=True)
    plt.suptitle('Openmax | %s' % metric)

    global_ax = plt.subplot(222)
    global_ax.set_ylabel('known classes')
    global_ax.set_xlabel('best epsilon')
    global_ax.grid(ls=":")
    global_ax.spines['top'].set_visible(False)
    global_ax.spines['right'].set_visible(False)


    for config_idx, (kkc, uuc) in enumerate(tqdm(config)):
        
        kkc_idx = kkc.shape[0]
        uuc_idx = uuc.shape[0]
        
        toss = config_idx % 5
        
        scores = results[metric_idx,config_idx].mean(0)

        # Analysis part
        if toss == 0:
            outer_accumulator = []
            
        outer_accumulator.append(scores.numpy())
        
        if toss == repeats-1:
            outer_accumulator = np.array(outer_accumulator)
            outer_accumulator = np.mean(outer_accumulator, axis=0)
            
            outer_accumulator = gaussian_filter(outer_accumulator, sigma=1)
            
            aa = ax[11-kkc_idx-2, uuc_idx-1]        
            im = aa.imshow(outer_accumulator, aspect='auto', vmax=1, vmin=0, cmap=cmaps[metric_idx])
            step = 3
            
            aa.set_yticks(np.arange(0,q,step), 
                        ['%.2f' % s for s in epsilons.numpy()[::step]])
            aa.set_xticks(np.linspace(0,epochs, 4).astype(int))

            if uuc_idx-1 == 0:
                aa.set_ylabel('%i known' % kkc_idx)
            if kkc_idx-2 == 0:
                aa.set_xlabel('%i unknown' % uuc_idx)
            
            best_epsilon_mask = outer_accumulator == np.max(outer_accumulator)
                
            much_epsilons = np.sum(best_epsilon_mask, axis=1)
            
            best_epsilons = epsilons[much_epsilons > 0]
            much_epsilons = much_epsilons[much_epsilons > 0]            
            
            global_ax.scatter(best_epsilons, 
                    [kkc_idx for v in best_epsilons],
                    s=(200*much_epsilons/np.sum(much_epsilons)), #10+uuc_idx*30, 
                    #alpha=much_epsilons/np.sum(much_epsilons)[:3],
                    color=cmaps_[metric_idx])

            
            global_ax.set_xlim(0,1)
            
            if config_idx==4:
                cb_ax = fig.add_axes([.92, 0.1, 0.02, 0.82])
                cbar = fig.colorbar(im, cax=cb_ax)
                plt.subplots_adjust(top=0.92)
     
    for i in range(8):
        for j in range(8):
            if i < j:
                try:
                    fig.delaxes(ax[i][j])
                except:
                    pass
                           
    plt.savefig('foo.png')
    plt.savefig('figures/openmax-calibration-%i.png' % metric_idx)