import numpy as np
import matplotlib as mpl; mpl.use('agg')
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable

class Additions(nn.Module):
    def __init__(self):
        super(Additions, self).__init__()
        self.layer1 = nn.Linear(2, 1, bias=False)

    def forward(self, x):
        out = self.layer1(x)
        return out

class NLAdditions(nn.Module):
    def __init__(self):
        super(NLAdditions, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(4,10),
            nn.ReLU(),
            nn.Linear(10,10),
            nn.ReLU(),
            nn.Linear(10,3))

    def forward(self, x):
        out = self.layer1(x)
        return out

if __name__ == '__main__':
    lres = np.load('../xmatch/lres_ml.npy')    
    hres = np.load('../xmatch/hres_ml.npy')    

    model = NLAdditions()

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    total_loss = []
    num_samples = 100

    for l,h in tqdm(zip(lres, hres)):
        lt = torch.tensor(l)
        ht = torch.tensor(h)
        input_data, target = Variable(lt).float(), Variable(ht).float()
        
        output = model(input_data)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss.append(loss)
        print(loss)
