import torch
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
from torcheval.metrics import BinaryConfusionMatrix
from train import *
import os
def rnaglib_measure_performance_during_training(max_eps, save_dir, net, net_name, train_loader, val_loader, test_loader, num_iters):
    """
    max_eps: number of epochs to train for
    net: input neural network
    net_name: a descriptive string which the network perfromance data will be saved under
    data: training data
    val_percent: the percentage of the training data we wish to use for the validaiton set
    data pdbs: the pdbs of the entries in the dataset in order
    test_data: test set data
    test_pdbs: test set data pdbs in order
    sites_dict: binding sites dictionary
    num_iters: the number of pooling iterations to run on each batch

    runs a training loop for "max_eps" epochs on "data"
    evaluates performance on valdiation and test sets every 5 epochs and saves the networks with the best performance
    """
    if not os.path.isdir(save_dir):
        print("could not find save directory")
        return -1
    best_MCC_val = -100
    val_MCC_overtime = []
    val_AUC_overtime = []
    val_conf_mats = []

    best_MCC_test = -100
    test_MCC_overtime = []
    test_AUC_overtime = []
    test_conf_mats = []
    
    epochs = []
    train_loss_overtime = []

    val_targets = [data['graph'].y.detach() for data in val_loader]
    val_targets = torch.cat(val_targets).long()
    test_targets = [data['graph'].y.detach() for data in test_loader]
    test_targets = torch.cat(test_targets).long()
    for epoch in range(0, max_eps+1):
        print("EPOCH", epoch)
        #print(net.poolLayer.lin.weight)
        #print(net.poolLayer.lin2.weight)
        net.train()
        train_loss = train_RGCNPool_net_BCE(net, train_loader, num_iters)
        #if True:
        train_loss_overtime.append(train_loss)
        if epoch%5==0:
            print("TESTING RESULTS")
            epochs.append(epoch)
            outs = get_x_scores(net, val_loader, num_iters)
            outs = torch.cat(outs)
            outs = outs.squeeze(1)
            #print(outs[:20])
            AUC =  roc_auc_score(val_targets, outs)
            print("val AUC:", AUC)
            outs = torch.round(outs).long()
            metric = BinaryConfusionMatrix()
            metric.update(outs, val_targets)
            res = metric.compute()
            print(res)
            MCC = matthews_corrcoef(val_targets, outs)
            print("val MCC:", MCC)

            if MCC > best_MCC_val:
                torch.save(net.state_dict(), save_dir+net_name+'_best_val_performance.pt')
                best_MCC_val = MCC
            val_MCC_overtime.append(MCC)
            val_AUC_overtime.append(AUC)
            val_conf_mats.append(res)

            outs = get_x_scores(net, test_loader, num_iters)
            outs = torch.cat(outs)
            outs = outs.squeeze(1)
            AUC =  roc_auc_score(test_targets, outs)
            print("test AUC:", AUC)
            outs = torch.round(outs).long()
            metric = BinaryConfusionMatrix()
            metric.update(outs, test_targets)
            res = metric.compute()
            print(res)
            MCC = matthews_corrcoef(test_targets, outs)
            print("test MCC:", MCC)
            test_MCC_overtime.append(MCC)
            test_AUC_overtime.append(AUC)
            test_conf_mats.append(res)
    return test_AUC_overtime, test_MCC_overtime, test_conf_mats, val_AUC_overtime, val_MCC_overtime, val_conf_mats, train_loss_overtime

path='./rnaglib_saved_networks_squareEdgeLabels/'
num_iters=3
loader_batch_size=64
train_loader=torch.load( "./data/datasets/rnaglib_train_all_6A_"+str(loader_batch_size)+".pt")
val_loader=torch.load( "./data/datasets/rnaglib_val_all_6A_"+str(loader_batch_size)+".pt")
test_loader=torch.load("./data/datasets/rnaglib_test_all_6A_"+str(loader_batch_size)+".pt")


for num_layers in [4]:
    for min_score in [0.75, 0.9]:
        for min_edge_score in [0.6, 0.75]:
            rand = RGCNPoolNet(644, num_layers, min_edge_score, min_score)
            net_name = 'RGCNx'+str(num_layers)+'_edgePool_MSE_'+str(min_edge_score)+'MinEdgescore_'+str(min_score)+'MinPercentile_percent_edge_labels_'+ str(num_iters)+'iters_noOnehop_fixed_DO0.1_WD0.0_ligands'
            dfs = rnaglib_measure_performance_during_training(200, path, rand, net_name, train_loader, val_loader, test_loader,  num_iters)
            torch.save(dfs, path+net_name+'_RESULTS')

