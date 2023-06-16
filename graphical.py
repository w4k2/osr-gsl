from torch.utils.data import DataLoader
import torchosr as osr
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from models.GSL import GSL
from torch import nn
from typing import Any, Tuple
import numpy as np
from torchvision.datasets import VisionDataset

class Make_classification_base(VisionDataset):
    def __init__(
        self,
        X, y, 
        root: str
    ) -> None:
        super().__init__(root)

        self.data, self.targets = torch.tensor(X).to(torch.float32), y

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        obj, target = self.data[index], int(self.targets[index])
        return obj, target

    def __len__(self) -> int:
        return len(self.targets)
    
    def _n_classes(self) -> int:
        return len(np.unique(self.targets))

def ls():
    return nn.Sequential(
        nn.Linear(2, 10),
        nn.ReLU(),
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32,64),
        nn.ReLU(),
        )

X, y = make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=3, n_clusters_per_class=1)
base = Make_classification_base(root='data', X=X, y=y)

kkc = [1, 2]
uuc = [0]

train, test = osr.data.get_train_test(base, kkc, uuc, root='data', tunning=False, fold=1, seed=233)

train_data_loader = DataLoader(train, batch_size=64, shuffle=True)
test_data_loader = DataLoader(test, batch_size=64, shuffle=True)


# Initialize models
models = [osr.models.TSoftmax(
            n_known=len(kkc), 
            lower_stack=ls(), 
            epsilon=osr.utils.get_softmax_epsilon(len(kkc))),
          osr.models.Openmax(
            n_known=len(kkc), 
            lower_stack=ls(), 
            epsilon=osr.utils.get_openmax_epsilon(len(kkc))),
          GSL(n_known=len(kkc), 
                lower_stack=ls(), 
                sigma=3., n_generated=.5)]

epochs = 10
lr = [1e-3, 1e-3, 1e-2]
res = np.zeros((3, 4, epochs))

# TRAIN
for m_id, m in enumerate(models):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer_gsl = torch.optim.SGD(m.parameters(), lr=1e-2)

    for epoch in range(epochs):
        m.train(train_data_loader, loss_fn, optimizer_gsl)
        scores = m.test(test_data_loader, loss_fn)
        res[m_id, :, epoch] = torch.tensor(scores)
        print(res[m_id, :, epoch])


# TEST ON SCATTER
mn = 2
space_x = np.linspace(mn*np.min(X[:,0]),mn*np.max(X[:,0]),150)
space_y = np.linspace(mn*np.min(X[:,1]),mn*np.max(X[:,1]),150)
space = torch.tensor(np.array([[x,y] for x in space_x for y in space_y])).to(torch.float32)

print(space.shape)
print(X)
print(space)

labels = ['TSM', 'OM', 'GSL']
cols = np.array(plt.cm.jet(np.linspace(0.2,0.8,3)))
fig, ax = plt.subplots(1,3,figsize=(12,4))    

for m_id, m in enumerate(models):
    aa = m.predict(space)
    
    print(np.unique(aa, return_counts=True))
    
    ax[m_id].scatter(space[:,0], space[:,1], c=cols[aa], alpha=0.02)
    ax[m_id].scatter(X[:,0], X[:,1], c=cols[y])
    ax[m_id].set_title(labels[m_id])

plt.tight_layout()
plt.savefig('foo.png')


    

    