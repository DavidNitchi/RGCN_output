#import sys
#sys.path.insert(0,'/Users/davidnitchi/Downloads/research_2024/RGCN_output')
from RGCNPool import *
import torch.optim as optim

#makes tensor where the edge labels are (total number positive residues)/(total number of residues) joined by the edge
# there can be more than 2 residues joioned by an edge due to pooling
def make_edge_labels_percents(edge_index, y):
    new_labels = []
    for ind in range(0, edge_index.shape[1]):
        new_labels.append((y[edge_index[0][ind]]+y[edge_index[1][ind]])/2)
    return torch.FloatTensor(new_labels)

def make_edge_labels_percents_y2d(edge_index, y_2d):
    new_labels = []
    for ind in range(0, edge_index.shape[1]):
        new_labels.append((y_2d[edge_index[0][ind]][0]+y_2d[edge_index[1][ind]][0])/(y_2d[edge_index[0][ind]][1]+y_2d[edge_index[1][ind]][1]))
    return torch.FloatTensor(new_labels)

class RGCNPoolLoss(torch.nn.Module):
    def __init__(self):
        super(RGCNPoolLoss, self).__init__()
        self.L1 = nn.L1Loss(reduction='none')
    def forward(self, outs, targets):
        inds = (targets == 1).nonzero(as_tuple=True)[0]
        #print(outs)
        #print(targets)
        L1_loss = self.L1(outs, targets)
        #returns L1 loss on edge scores, also add a term of extra loss on all positive edges,
        #this extra loss on positives was needed because there are many more negative than positive edges so the model would often end up never pooling anything
        return torch.sum(L1_loss)+torch.sum(torch.index_select(L1_loss, 0, inds)*2)
    
def train_RGCNPool_net(model, loader, num_iters):
    model.train()
    optimizer = optim.Adam(model.parameters())
    criterion = RGCNPoolLoss()
    #sig = nn.Sigmoid()
    total_loss = 0.0
    for data in loader:
        #print(data)
        y = data.y
        ones = torch.ones(y.shape)
        y_2d = torch.column_stack([y, ones])
        targets = make_edge_labels_percents_y2d(data.edge_index, y_2d)
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch
        for _ in range(0, num_iters):
            optimizer.zero_grad()
            x, outs, pool_info, y_2d, edge_index, edge_attr, batch, x_scores_RGCN  = model(x, edge_index, edge_attr, batch, y_2d)
            edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
            
            print('number of pooled edges', torch.min((pool_info.new_edge_score == 100).nonzero(as_tuple=True)[0]))
            print('number of total edges', edge_index.shape[1])
            loss = criterion(outs, targets)
            total_loss += loss
            loss.backward()
            optimizer.step()
            targets = make_edge_labels_percents_y2d(edge_index, y_2d)
        #break
    print('this epochs total loss:', total_loss)
    return total_loss