from jaccard import *
from train import *
import numpy as np
import pandas as pd
import pickle
import random
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
from torcheval.metrics import BinaryConfusionMatrix

def measure_performance_during_training(max_eps, net, net_name, data, val_data_percent, data_pdbs, test_data, test_pdbs, sites_dict, num_iters):
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
    #val_inds = torch.load('./validation_indices.pt')
    test_best_performance = []
    test_biggest_performance = []
    val_performance = []
    val_biggest_performance = []
    val_data = data[-int(len(data)*val_data_percent):]
    val_pdbs = data_pdbs[-int(len(data)*val_data_percent):]
    train_data = data[:-int(len(data)*val_data_percent)]
    best_MCC = -100
    MCC_overtime = []
    AUC_overtime = []
    conf_mats = []
    #train_data = data
    #val_data = [data[i] for i in val_inds]
    #val_pdbs = [data_pdbs[i] for i in val_inds]
    #train_data = [data[i] for i in range(0, len(data)) if i not in val_inds]
    print(len(train_data))
    print(len(val_data))
    train_loader = DataLoader(train_data, batch_size =64, shuffle = True)
    epochs = []
    train_loss_overtime = []
    biggest_val_perf = 0
    biggest_test_perf = 0
    best_val_perf = 0
    best_test_perf = 0
    targets = [data.y.detach() for data in val_data]
    targets = torch.cat(targets).long()
    for epoch in range(0, max_eps+1):
        print('EPOCH', epoch)
        #print(net.poolLayer.lin.weight)
        #print(net.poolLayer.lin2.weight)
        net.train()
        train_loss = train_RGCNPool_net(net, train_loader, num_iters)
        #if True:
        train_loss_overtime.append(train_loss)
        if epoch%5==0:
            epochs.append(epoch)
            outs = get_x_scores(net, val_data, num_iters)
            outs = torch.cat(outs)
            outs = outs.squeeze(1)
            AUC =  roc_auc_score(targets, outs)
            print("AUC:", AUC)
            outs = torch.round(outs).long()
            metric = BinaryConfusionMatrix()
            metric.update(outs, targets)
            res = metric.compute()
            print(res)
            MCC = matthews_corrcoef(targets, outs)
            print("MCC:", MCC)

            if MCC > best_MCC:
                torch.save(net.state_dict(), './rnaglib_nr_saved_networks/'+net_name+'_best_val_performance.pt')
                best_MCC = MCC
            MCC_overtime.append(MCC)
            AUC_overtime.append(AUC)
            conf_mats.append(res)


            #val_gs = get_groups_dict(net, val_data, val_pdbs, val_pdbs, sites_dict, 3)
            # val_1_jccrd = plot_pockets_jaccard(val_gs, 1)
            # val_3_jccrd = plot_pockets_jaccard(val_gs, 3)

            # val_m1 = np.median(val_1_jccrd)
            # val_m3 = np.median(val_3_jccrd)
            # val_performance.append([val_m1, val_m3])

            #if val_m1 > best_val_perf:
                #torch.save(net.state_dict(), './saved_networks_Val4/'+net_name+'_best_val_performance.pt')
                #best_val_perf = val_m1
            
            #test_gs = get_groups_dict(net, test_data, test_pdbs, test_pdbs, sites_dict, 3)
            # test_1_jccrd = plot_pockets_jaccard(test_gs, 1)
            # test_3_jccrd = plot_pockets_jaccard(test_gs, 3)

            # test_m1 = np.median(test_1_jccrd)
            # test_m3 = np.median(test_3_jccrd)

            # if test_m1 > best_test_perf:
            #     torch.save(net.state_dict(), './saved_networks_Val4/'+net_name+'_best_test_performance.pt')
            #     best_test_perf = test_m1
            
            #print(prec_test, rec_test, MCC_test)
            
            #test_best_performance.append([test_m1, test_m3])
            

            # val_1_jccrd = plot_biggest_pockets_jaccard(val_gs, 1)
            # val_3_jccrd = plot_biggest_pockets_jaccard(val_gs, 3)

            # val_m1 = np.median(val_1_jccrd)
            # val_m3 = np.median(val_3_jccrd)

            # #if val_m1 > biggest_val_perf:
            #     #torch.save(net.state_dict(), './saved_networks_Val4/'+net_name+'_biggest_val_performance.pt')
            #     #biggest_val_perf = val_m1

            # test_1_jccrd = plot_biggest_pockets_jaccard(test_gs, 1)
            # test_3_jccrd = plot_biggest_pockets_jaccard(test_gs, 3)

            # test_m1 = np.median(test_1_jccrd)
            # test_m3 = np.median(test_3_jccrd)
            
            # # if test_m1 > biggest_test_perf:
            # #     torch.save(net.state_dict(), './saved_networks_Val4/'+net_name+'_biggest_test_performance.pt')
            # #     biggest_test_perf = test_m1

            # test_biggest_performance.append([test_m1, test_m3])
            # val_biggest_performance.append([val_m1, val_m3])
            #data_performance.append([train_loss.item()])
    #num_eps = max_eps
    
    #test_best_df = pd.DataFrame(test_best_performance, index=epochs, columns=['Jaccard_top_1', 'Jaccard_top_3'])
    test_biggest_df = pd.DataFrame(test_biggest_performance, index=epochs, columns=['Jaccard_Largest_1', 'Jaccard_Largest_3'])

    # #val_best_df = pd.DataFrame(val_performance, index=epochs, columns=['Jaccard_top_1', 'Jaccard_top_3'])
    val_biggest_df = pd.DataFrame(val_biggest_performance, index=epochs, columns=['Jaccard_largest_1', 'Jaccard_largest_3'])

    
    #return test_best_df, test_biggest_df, val_best_df, val_biggest_df, train_loss_overtime
    return test_biggest_df, val_biggest_df, train_loss_overtime, MCC_overtime, AUC_overtime, conf_mats

all_ligands_data_4A = torch.load('./data/datasets/train_data_ligands_4A_embeddings_min4nbs_0.3seqId.pt')
#all_ligands_ions_data_4A = torch.load('./data/datasets/g_data_ligands_ions_4A_embeddings_min4nbs_0.3seqId.pt')

used_pdbs_ligands_4A = torch.load('./data/datasets/train_pdbs_ligands_4A_embeddings_min4nbs_0.3seqId.pt')
#used_pdbs_ligands_ions_4A = torch.load('./data/datasets/pdbs_ligands_ions_4A_embeddings_min4nbs_0.3seqId.pt')

test_set_ligands_4A = torch.load('./data/datasets/test_data_ligands_4A_embeddings_min4nbs_0.3seqId.pt')
#test_set_ligands_ions_4A = torch.load('./data/datasets/g_data_ligands_ions_4A_embeddings_min4nbs_0.3seqId_testSet.pt')

used_pdbs_ligands_test_set_4A = torch.load('./data/datasets/test_pdbs_ligands_4A_embeddings_min4nbs_0.3seqId.pt')
#used_pdbs_ligands_ions_test_set_4A = torch.load('./data/datasets/pdbs_ligands_ions_4A_embeddings_min4nbs_0.3seqId_testSet.pt')

used_pdbs_4A = torch.load('./data/datasets/pdbs_ligands_ions_4A_embeddings_min4nbs_0.3seqId.pt')
sites_dict_4A = pickle.load(open('./data/Hariboss+PDBbind+RCSB_data_4A_recent.pkl', 'rb'))

RGCN_data = []
for data in all_ligands_data_4A:
    RGCN_data.append(Data(x=data.x, y=data.y, edge_attr = transform_edge_attr(data.edge_attr), edge_index = data.edge_index))
test_4A_RGCN = []
for data in test_set_ligands_4A:
    test_4A_RGCN.append(Data(x=data.x, y=data.y, edge_attr = transform_edge_attr(data.edge_attr), edge_index = data.edge_index))

RGCN_ions = []
for data in all_ligands_data_4A:
    RGCN_ions.append(Data(x=data.x, y=data.y, edge_attr = transform_edge_attr(data.edge_attr), edge_index = data.edge_index))
test_4A_ions_RGCN = []
for data in test_set_ligands_4A:
    test_4A_ions_RGCN.append(Data(x=data.x, y=data.y, edge_attr = transform_edge_attr(data.edge_attr), edge_index = data.edge_index))



num_iters=3
for num_layers in [2, 3, 4]:
    for min_score in [0.9]:
        rand = RGCNPoolNet(640, num_layers, 0, min_score)
        dfs = measure_performance_during_training(50, rand, 'RGCNx'+str(num_layers)+'_edgePool_L1avg_'+str(min_score)+'Minscore_percent_edge_labels_'+ str(num_iters)+'iters_noOnehop_fixed_DO0.1_ligands', RGCN_ions[:], 0.15, used_pdbs_ligands_4A[:], test_4A_ions_RGCN, used_pdbs_ligands_test_set_4A, sites_dict_4A, num_iters)
        torch.save(dfs, './rnaglib_nr_saved_networks/RGCNx'+str(num_layers)+'_edgePool_L1avg_'+str(min_score)+'Minscore_percent_edge_labels_'+ str(num_iters)+'iters_noOnehop_fixed_DO0.1_ligands_RESULTS')
