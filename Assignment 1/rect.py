# rect.py

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Network(torch.nn.Module):
    def __init__(self, layer, hid):
        super(Network, self).__init__()
        self.layer = layer
        self.hlayer_1 = nn.Linear(2, hid)
        self.hlayer_2 = nn.Linear(hid, hid)
        self.out = nn.Linear(hid, 1)
        # INSERT CODE HERE


    def forward(self, input):
        input = input.view(input.shape[0], -1)
        hlayer_1 = self.hlayer_1(input)
        hlayer_1 = torch.tanh(hlayer_1)
        self.hid_1_layer = hlayer_1

        if (self.layer == 1):
            out = self.out(hlayer_1)
            out = torch.sigmoid(hlayer_1)
        else:
            hlayer_2 = self.hlayer_2(hlayer_1)
            hlayer_2 = torch.tanh(hlayer_2)
            self.hid_2_layer = hlayer_2
            out = self.out(hlayer_2)
            out = torch.sigmoid(hlayer_2)
        
        # CHANGE CODE HERE

        return out

def graph_hidden(net, layer, node):
    xrange = torch.arange(start=-8,end=8.1,step=0.01,dtype=torch.float32)
    yrange = torch.arange(start=-8,end=8.1,step=0.01,dtype=torch.float32)
    xcoord = xrange.repeat(yrange.size()[0])
    ycoord = torch.repeat_interleave(yrange, xrange.size()[0], dim=0)
    grid = torch.cat((xcoord.unsqueeze(1),ycoord.unsqueeze(1)),1)

    with torch.no_grad(): # suppress updating of gradients
        net.eval()        # toggle batch norm, dropout
        output = net(grid)
        net.train() # toggle batch norm, dropout back again
        if (layer ==1):
            pred = (net.hid_1_layer[:,node] >= 0).float()
        else:
            pred = (net.hid_2_layer[:,node] >= 0).float()
           

        # plot function computed by model
        plt.clf()
        plt.pcolormesh(xrange,yrange,pred.cpu().view(yrange.size()[0],xrange.size()[0]),
                       cmap='Wistia', shading='auto')
    # INSERT CODE HERE
