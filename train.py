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
    return torch.square(torch.FloatTensor(new_labels))

def make_edge_labels_percents_y2d(edge_index, y_2d):
    new_labels = []
    for ind in range(0, edge_index.shape[1]):
        new_labels.append((y_2d[edge_index[0][ind]][0]+y_2d[edge_index[1][ind]][0])/(y_2d[edge_index[0][ind]][1]+y_2d[edge_index[1][ind]][1]))
    return torch.square(torch.FloatTensor(new_labels))

class RGCNPoolLoss(torch.nn.Module):
    def __init__(self):
        super(RGCNPoolLoss, self).__init__()
        self.L1 = nn.MSELoss(reduction='mean')
    def forward(self, outs, targets):
        #inds = (targets >= 0.75).nonzero(as_tuple=True)[0]
        #print(outs)
        print(sum(outs)/outs.shape[0])
        #print(targets)
        #L1_loss = self.L1(outs, targets)
        #returns L1 loss on edge scores, also add a term of extra loss on all positive edges,
        #this extra loss on positives was needed because there are many more negative than positive edges so the model would often end up never pooling anything
        #return (torch.sum(L1_loss)+torch.sum(torch.index_select(L1_loss, 0, inds)*2))/L1_loss.shape[0]
        return self.L1(outs, targets)
def train_RGCNPool_net(model, loader, num_iters):
    model.train()
    optimizer = optim.Adam(model.parameters())
    criterion = RGCNPoolLoss()
    #sig = nn.Sigmoid()
    total_loss = 0.0
    counter = 0
    for data in loader:
        counter += 1
        print("batch", counter)
        #print(data)
        #this line is needed when using rnaglib data loader
        data=data['graph']
        y = data.y
        ones = torch.ones(y.shape)
        y_2d = torch.column_stack([y, ones])
        targets = make_edge_labels_percents_y2d(data.edge_index, y_2d)
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch
        for i in range(0, num_iters):
            print("iteration", i)
            optimizer.zero_grad()
            x, outs, pool_info, y_2d, edge_index, edge_attr, batch, x_scores_RGCN, edgePool_x  = model(x, edge_index, edge_attr, batch, y_2d)
            #print(outs)
            edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
            if (pool_info.new_edge_score == 1).nonzero(as_tuple=True)[0].shape[0] > 0:
                print('number of pooled edges', torch.min((pool_info.new_edge_score == 1).nonzero(as_tuple=True)[0]))
            else:
                print('All edges pooled', pool_info.new_edge_score.shape[0])
            print('number of total edges', edge_index.shape[1])
            loss = criterion(outs, targets)
            total_loss += loss
            loss.backward()
            optimizer.step()
            targets = make_edge_labels_percents_y2d(edge_index, y_2d)
        #break
    print('this epochs total loss:', total_loss)
    return total_loss

def train_RGCNPool_net_BCE(model, loader, num_iters):
    model.train()
    optimizer = optim.Adam(model.parameters())
    criterion = RGCNPoolLoss()
    sig = nn.Sigmoid()
    #sig = nn.Sigmoid()
    total_loss = 0.0
    counter = 0
    for data in loader:
        counter += 1
        print("batch", counter)
        #print(data)
        #this line is needed when using rnaglib data loader
        data=data['graph']
        y = data.y
        ones = torch.ones(y.shape)
        y_2d = torch.column_stack([y, ones])
        targets = make_edge_labels_percents_y2d(data.edge_index, y_2d)
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch
        unpools = []
        for i in range(0, num_iters):
            print("iteration", i)
            optimizer.zero_grad()
            x, outs, pool_info, y_2d, edge_index, edge_attr, batch, x_scores_RGCN, edgePool_x  = model(x, edge_index, edge_attr, batch, y_2d)
            #print(outs)
            unpools.append(pool_info)
            edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
            # if (pool_info.new_edge_score == 1).nonzero(as_tuple=True)[0].shape[0] > 0:
            #     print('number of pooled edges', torch.min((pool_info.new_edge_score == 1).nonzero(as_tuple=True)[0]))
            # else:
            #     print('All edges pooled', pool_info.new_edge_score.shape[0])
            # print('number of total edges', edge_index.shape[1])

        # tmp_x = edgePool_x
        # for unpool in reversed(unpools):
        #     #print(unpool.new_edge_score[:10])
        #     new_x, edge_ind, batch = model.poolLayer.unpool(tmp_x, unpool)
        #     #print(new_x[:10])
        #     tmp_x = new_x
            loss = criterion(outs, targets)
            total_loss += loss
            loss.backward()
            optimizer.step()
            targets = make_edge_labels_percents_y2d(edge_index, y_2d)

    print('this epochs total loss:', total_loss)
    return total_loss
def test_RGCNPool_net(model, data, num_iters):
    model.eval()
    model.RGCN.eval()
    model.poolLayer.eval()
    #with torch.no_grad():
        #optimizer = optim.Adam(model.parameters())
        #criterion = nn.L1Loss(reduction = 'sum')
        #sig = nn.Sigmoid()
    pool_info = []
    total_loss = 0.0
        #for data in loader:
            #print(data)
    y = data.y
    #print(y)
    ones = torch.ones(y.shape)
    y_2d = torch.column_stack([y, ones])
    targets = make_edge_labels_percents(data.edge_index,data.y)
    x = data.x
    edge_index = data.edge_index
    edge_attr = data.edge_attr
    batch = data.batch
    for i in range(0, num_iters):
            #optimizer.zero_grad()
        x, outs, tmp_pool_info, y_2d, edge_index, edge_attr, batch, x_scores_RGCN, edgePool_x  = model(x, edge_index, edge_attr, batch, y_2d)
            #print("x scores after pooling")
            #print(x_scores_RGCN)
            #print("++++++++++++++++++++++++")
            #print(outs)
            #print("====== End of occ ======")
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
                #print('number of pooled edges', torch.min((pool_info[-1].new_edge_score == 100).nonzero(as_tuple=True)[0]))
                #print('number of total edges', edge_index.shape[1])
                #print(outs.shape)
                 #the data.y is the y_2d value
                #print(sig(outs))
                #print(targets)
        pool_info.append(tmp_pool_info)
            #loss = criterion(outs, targets)
            #total_loss += loss
            #loss.backward()
            #optimizer.step()
        targets = make_edge_labels_percents_y2d(edge_index, y_2d)
        #print(y_2d)
            #break
    new_y = ""
    return x, pool_info, outs, new_y, y_2d, x_scores_RGCN, edgePool_x

def get_x_scores(model, dataloader, num_iters):
    x_scores = []
    sig = nn.Sigmoid()
    #dataloader = DataLoader(dataloader, 1)
    for d in dataloader:
        #print(d)
        #this line is needed when using rnaglib dataloader
        d=d['graph']
        #print(d)
        x, pool_info, outs, new_y, y_2d, x_scores_RGCN, edgePool_x = test_RGCNPool_net(model, d, num_iters)
        tmp_x = edgePool_x

        #print(len(pool_info))
        counter = 0
        tmp_x = torch.zeros_like(tmp_x)
        for unpool in reversed(pool_info):
            counter += 1
            #print(unpool.new_edge_score[:10])
            new_x, edge_ind, batch = model.poolLayer.unpool(tmp_x, unpool, counter)
            #print(new_x[:10])
            tmp_x = new_x
        #print(tmp_x[:3])
        #tmp_x = tmp_x-torch.mean(tmp_x)
        ##x_scores.append(sig(tmp_x).detach())

        x_scores.append(tmp_x)
    #print(len(x_scores))
    #print(x_scores[0].shape)
    #print(x_scores)
    return x_scores