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
    def __init__(self, xlist, ylist):
        super(NLAdditions, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(len(xlist),10),
            nn.ReLU(),
            nn.Linear(10,10),
            nn.ReLU(),
            nn.Linear(10,len(ylist)))
        
        self.xmean = np.mean(xlist, axis=0)
        self.xstd = np.std(xlist, axis=0)
        self.ymean = np.mean(ylist, axis=0)
        self.ystd = np.mean(ylist, axis=0)

        self.x, self.y = self.normalize(xlist, ylist)

    def forward(self, x):
        out = self.layer1(x)
        return out
    
    def normalize(self, xlist, ylist):
        x = np.subtract(xlist, self.xmean)
        x = np.divide(x, self.xstd)
        y = np.subtract(ylist, self.ymean)
        y = np.divide(y, self.ystd)
        return x, y
    
    def unnormalize(self, xlist, ylist):
        x = np.multiply(xlist, self.xstd)
        x = np.add(x, self.xmean)
        y = np.multiply(ylist, self.ystd)
        y = np.add(y, self.ymean)
        return x, y


if __name__ == '__main__':
    lres = np.load('../xmatch/lres_ml.npy')    
    hres = np.load('../xmatch/hres_ml.npy')    

    model = NLAdditions(lres, hres)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    total_loss = []
    num_samples = 100

    for l,h in tqdm(zip(model.x, model.y)):
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
