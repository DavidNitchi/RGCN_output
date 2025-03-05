import torch
from jaccard import *
import matplotlib.pyplot as plt
import numpy as np
import pickle
from Bio.PDB import PDBParser, NeighborSearch
import pandas as pd
import os
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
from torcheval.metrics import BinaryConfusionMatrix
all_ligands_data_4A = torch.load('./../g_data_ligands_4A_embeddings_min4nbs_0.3seqId.pt')
#all_ligands_data_onehop_4A = torch.load('./../g_data_ligands_4A_onehop_embeddings_min4nbs_0.3seqId.pt')
#used_pdbs_ligands_onehop_4A = torch.load('./../pdbs_ligands_4A_onehop_embeddings_min4nbs_0.3seqId.pt')
used_pdbs_ligands_4A = torch.load('./../pdbs_ligands_4A_embeddings_min4nbs_0.3seqId.pt')
sites_dict_4A = pickle.load(open('./../Hariboss+PDBbind+RCSB_data_4A_recent.pkl', 'rb'))
test_set_ligands_4A = torch.load('./../g_data_ligands_4A_embeddings_min4nbs_0.3seqId_testSet.pt')
used_pdbs_ligands_test_set_4A = torch.load('./../pdbs_ligands_4A_embeddings_min4nbs_0.3seqId_testSet.pt')
RB9_pyg = torch.load('./../RB9_data_NodeEmbeddings.pt')
#print(used_pdbs_ligands_test_set_4A)


RGCN_data = []
for data in all_ligands_data_4A:
    RGCN_data.append(Data(x=data.x, y=data.y, edge_attr = transform_edge_attr(data.edge_attr), edge_index = data.edge_index))

test_4A_RGCN = []
for data in test_set_ligands_4A:
    test_4A_RGCN.append(Data(x=data.x, y=data.y, edge_attr = transform_edge_attr(data.edge_attr), edge_index = data.edge_index))
RB9_RGCN = []
for data in RB9_pyg:
    RB9_RGCN.append(Data(x=data.x, y=data.y, edge_attr = transform_edge_attr(data.edge_attr), edge_index = data.edge_index))

val_data_percent = 0.15
#print(-int(len(RGCN_data)*val_data_percent))
val_data = RGCN_data[-int(len(RGCN_data)*val_data_percent):]
val_pdbs = used_pdbs_ligands_4A[-int(len(RGCN_data)*val_data_percent):]

#good_pdbs = ['1Q8N', '2JUK', '2TOB', '1F1T', '1FMN', '430D'
ligands = 'ligands'
real_chains_TE18 = {
    "2pwt": ({'pdb_chain':'A'}, {ligands: [("LHA",3)]}), 
    
    "5v3f": ({'pdb_chain':'A'}, {ligands: [("74G",1)]}), 
    
    "379d": ({'pdb_chain':'B'}, {ligands: [("CO",4)]}),
    
    "5bjo": ({'pdb_chain':'E'}, {ligands: [("MG",2), ('747', 1)]}), 
    
    "4pqv": ({'pdb_chain':'A'}, {ligands: [("MG",7)]}), 
    
    "430d": ({'pdb_chain':'A'}, {ligands: [("MG",9)]}), 
    
    "1nem": ({'pdb_chain':'A'}, {ligands: [("BDR",1), ("BDG",1), ("NEB",1), ("IDG",1)]}), 
    
    "1q8n": ({'pdb_chain':'A'}, {ligands: [("MGR",1)]}), 
    
    "1f1t": ({'pdb_chain':'A'}, {ligands: [("ROS",1)]}), 
    
    "2juk": ({'pdb_chain':'A'}, {ligands: [("G0B",1)]}), 
    
    "4yaz": ({'pdb_chain':'R'}, {ligands: [("4BW",1), ("MG", 4)]}), 
    
    "364d": ({'pdb_chain':'C'}, {ligands: [("MG",8)]}),
    
    "6ez0": ({'pdb_chain':'A'}, {ligands: [("U37",5)]}), 
    
    "2tob": ({'pdb_chain':'A'}, {ligands: [("TOA",1), ("2TB",1), ("TOC",1)]}), 
    
    "1ddy": ({'pdb_chain':'A'}, {ligands: [("B12",1), ("NME",1)]}),
    
    "1fmn": ({'pdb_chain':'A'}, {ligands: [("FMN",1)]}), 
    
    "2mis": ({'pdb_chain':'A'}, {ligands: [("MG",4)]}), 
    
    "4f8u": ({'pdb_chain':'B'}, {ligands: [("SIS",2)]}), 
    
    }
# small_pdbs = []
# big_pdbs = []
# small_data = []
# big_data = []
# for ind, d in enumerate(test_4A_RGCN):
#     if d.y.shape[0] < 40:
#         small_data.append(d)
#         small_pdbs.append(used_pdbs_ligands_test_set_4A[ind])
#     else:
#         big_data.append(d)
#         big_pdbs.append(used_pdbs_ligands_test_set_4A[ind])
#print(big_data)
"""
l_size = []
l_pos = []
for d in test_4A_RGCN:
    l_size.append(d.y.shape[0])
    l_pos.append(torch.sum(d.y))
size_ticks = [int(x) for x in np.linspace(0, 500, 25, endpoint=False)]
plt.figure(figsize=(15, 4))
plt.xticks(size_ticks)
plt.hist(l_size, bins=50)
#plt.show()
plt.hist(l_pos, bins=20)
plt.xticks(range(0, 50, 5))
plt.show()
"""

for num_layers in [4]:
    for min1 in [0.5, 0.75, 0.9]:
        for min_edge_score in [0.5, 0.6, 0.75]:
            perf = torch.load('./rnaglib_nr_saved_networks/RGCNx'+str(num_layers)+'_edgePool_MSE_'+str(min_edge_score)+'MinEdgeScore_'+str(min1)+'MinPercentile_percent_edge_labels_3iters_noOnehop_fixed_DO0.1_WD0.0_ligands_RESULTS')
            print("VAL overtime for "+str(min1)+","+str(min_edge_score))
            val_list = perf[4]
            test_list = perf[1]
            max_val = max(val_list)
            idx_max = val_list.index(max_val)
            print(idx_max)
            print(max_val,test_list[idx_max])
            train_loss = [t.detach().numpy() for t in perf[-1]]
            #plt.plot(train_loss, label=str(min_edge_score))
#plt.legend()
#plt.show()
# params = [(2, 0.25), (3, 0.25), (4, 0.9)]
# models = []
# all_scores = []
# for model_param in params:
#     print(model_param)
#     mod = RGCNPoolNet(model_param[0],0, model_param[1])
#     mod.load_state_dict(torch.load('./saved_networks_MCC/RGCNx'+str(model_param[0])+'_edgePool_L1avg_'+str(model_param[1])+'Minscore_percent_edge_labels_3iters_noOnehop_fixed_DO0.1_best_val_performance.pt'))
#     outs = get_x_scores(mod, TE18_RGCN, 3)
#     outs = torch.cat(outs)
#     outs = outs.squeeze(1)
#     all_scores.append(outs)
# joined = torch.vstack(all_scores)
# mean = torch.mean(joined, 0)

def getRnetOutputs(path_to_results_dir, pdbs):
    scores = []
    missing_pdbs = []
    for pdb in pdbs:
        g = all_graphs_dataset.get_pdbid(pdb)['rna']
        outputs = pd.read_csv(path_to_results_dir+'/'+pdb+'_result.csv')
        parser = PDBParser(QUIET=True)
        try:
            structure = parser.get_structure('protein', './../pdb_data/pdb_file/'+pdb+'.pdb')  
        except:
            missing_pdbs.append(pdb.upper())
        mod = list(structure.get_models())[0]
        residues = list(mod.get_residues())
        #print(residues)
        nodes = list(g.nodes())
        print(nodes)
        outs = []
        for res in residues:
            print("residue", res)
            try:
                ind = nodes.index(pdb.lower()+'.'+res.get_parent().id+'.'+str(res.id[1]))
            except:
                print("COULD NOT FIND RESIDUE", res)
                continue
            print(ind)
            #print('SCORE:', outputs.loc[ind, 'binding site probabilities'])
            outs.append(outputs.loc[ind, 'binding site probabilities'])
            #print(len(outs))
            #residues
        scores.append(torch.tensor(outs).detach())
    print(missing_pdbs)
    return torch.cat(scores)

def getRnetOutputsTE18(path_to_results_dir, real_chains_TE18):
    scores = []
    missing_pdbs = []

    for pdb in real_chains_TE18:
        chain_name = real_chains_TE18[pdb][0]['pdb_chain']
        g = all_graphs_dataset.get_pdbid(pdb)['rna']
        outputs = pd.read_csv(path_to_results_dir+'/'+pdb+chain_name+'_result.csv')
        parser = PDBParser(QUIET=True)
        try:
            structure = parser.get_structure('protein', './../pdb_data/pdb_file/'+pdb.upper()+'.pdb')   
        except:
            missing_pdbs.append(pdb.upper())
        mod = list(structure.get_models())[0]
        residues = list(mod.get_residues())
        #print(residues)
        nodes = list(g.nodes())
        print(nodes)
        outs = []
        for res in residues:
            print("residue", res)
             
            try:
                if res.get_parent().id != chain_name:
                    continue
                print("getting index of node:", pdb.lower()+'.'+res.get_parent().id+'.'+str(res.id[1]))
                ind = nodes.index(pdb.lower()+'.'+res.get_parent().id+'.'+str(res.id[1]))
            except:
                print("COULD NOT FIND RESIDUE", res)
                continue
            print(ind)
            #print('SCORE:', outputs.loc[ind, 'binding site probabilities'])
            outs.append(outputs.loc[ind, 'binding site probabilities'])
            #print(len(outs))
            #residues

        scores.append(torch.tensor(outs).detach())
    print(missing_pdbs)

    return torch.cat(scores)

# TE18_in_tes4A = [(ind ,pdb) for ind, pdb in enumerate(used_pdbs_ligands_test_set_4A) if pdb.lower() in TE18_json]
# TE18_in_test4A_pdbs = [data[1] for data in TE18_in_tes4A]
# TE18_in_test4A_inds = [data[0] for data in TE18_in_tes4A]
#good_pdbs = ['1Q8N', '2JUK', '2TOB', '1F1T', '1FMN', '430D']

#scores = getRnetOutputsTE18('~/Downloads/Rnetsite_Package_2.0/TE18_nativePDB/result', real_chains_TE18)
#print(scores)
#TE18_pdbs_indices = [TE18_json.index(x.lower()) for x in good_pdbs]
# print(TE18_json)
# print(TE18_in_tes4A)
# print(TE18_pdbs_indices)
line_styles = ['-', ':', '--', '-.']
# for ind, val in enumerate(TE18_in_tes4A):
#     print(val[1])
#     print(test_4A_RGCN[val[0]])
#     print(TE18_RGCN[TE18_pdbs_indices[ind]])

#targets = [data.y.detach() for ind, data in enumerate(TE18_RGCN) if ind in TE18_pdbs_indices]
# targets = [data.y.detach() for _, data in enumerate(TE18_RGCN)]

# targets = torch.cat(targets).long()

# print("AUC:", roc_auc_score(targets, scores))
# scores = torch.round(scores).long()
# metric = BinaryConfusionMatrix()
# metric.update(scores, targets)
# res = metric.compute()
# print(res)
# print("MCC:", matthews_corrcoef(targets, scores))
# print("Prec:", precision_score(targets, scores))
# print("Recall:", recall_score(targets, scores))

# print(os.path.isdir('./rnaglib_nr_saved_networks/'))
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(6, 12), gridspec_kw={'hspace': 0.4})

#ind_to_get = 1
# num_layers = 4
# for row_ind, ind_to_get in enumerate([4, 1]):
#     plot_counter = 0
#     for min_edge_sore in [0.5, 0.6, 0.75]:
#         #for min_score in [0.5, 0.75, 0.9]:
#             #RGCNx3_edgePool_L1avg_0.5Minscore_percent_edge_labels_3iters_noOnehop_fixed_DO0.1_RESULTS
#         min_pools = [0.5, 0.75, 0.9]
#         [min1, min2, min3] = min_pools
#         #RGCNx4_edgePool_MSE_0.75MinEdgescore_0.75MinPercentile_percent_edge_labels_3iters_noOnehop_fixed_DO0.1_WD0.0_ligands_RESULTS
#         perf1 = torch.load('./rnaglib_nr_saved_networks/RGCNx'+str(num_layers)+'_edgePool_MSE_'+str(min_edge_sore)+'MinEdgeScore_'+str(min1)+'MinPercentile_percent_edge_labels_3iters_noOnehop_fixed_DO0.1_WD0.0_ligands_RESULTS')
#         perf2 = torch.load('./rnaglib_nr_saved_networks/RGCNx'+str(num_layers)+'_edgePool_MSE_'+str(min_edge_sore)+'MinEdgeScore_'+str(min2)+'MinPercentile_percent_edge_labels_3iters_noOnehop_fixed_DO0.1_WD0.0_ligands_RESULTS')
#         perf3 = torch.load('./rnaglib_nr_saved_networks/RGCNx'+str(num_layers)+'_edgePool_MSE_'+str(min_edge_sore)+'MinEdgeScore_'+str(min3)+'MinPercentile_percent_edge_labels_3iters_noOnehop_fixed_DO0.1_WD0.0_ligands_RESULTS')
#         #perf2 = torch.load('./rnaglib_nr_saved_networks/RGCNx'+str(num_layers)+'_edgePool_MSE_'+str(min2)+'Minscore_percent_edge_labels_3iters_noOnehop_fixed_DO0.1_WD0.0_ligands_RESULTS')
#         #perf3 = torch.load('./rnaglib_nr_saved_networks/RGCNx'+str(num_layers)+'_edgePool_MSE_'+str(min3)+'Minscore_percent_edge_labels_3iters_noOnehop_fixed_DO0.1_WD0.0_ligands_RESULTS')
#         #perf4 = torch.load('./rnaglib_nr_saved_networks/RGCNx'+str(num_layers)+'_edgePool_MSE_'+str(min4)+'Minscore_percent_edge_labels_3iters_noOnehop_fixed_DO0.1_WD0.0_ligands_RESULTS')
#         for ind, perf in enumerate([perf1, perf2, perf3]):
#             axs[row_ind][plot_counter].plot(range(0, 21), [x for x in perf[ind_to_get]], linestyle =line_styles[ind], label = r"$minscore = %s$" %str(min_pools[ind]))
#         axs[row_ind][plot_counter].legend(fontsize=6)
#         axs[row_ind][plot_counter].set_xlabel("epoch")
#         axs[row_ind][plot_counter].set_ylabel("MCC")
#         axs[row_ind][plot_counter].set_title("MCC overtime on Test set with "+str(min_edge_sore)+" minimum edge score")
#         plot_counter += 1
# plt.show()
#print(len(val_data))
#"""

# for num_layers in [2, 3, 4]:
#     for min_score in [0.25, 0.5, 0.75, 0.9]:
#         #RGCNx3_edgePool_L1avg_0.5Minscore_percent_edge_labels_3iters_noOnehop_fixed_DO0.1_RESULTS
#         perf = torch.load('./saved_networks_MCC/RGCNx'+str(num_layers)+'_edgePool_L1avg_'+str(min_score)+'Minscore_percent_edge_labels_3iters_noOnehop_fixed_DO0.1_RESULTS')
#         print(num_layers, min_score)

#         #df = perf[0]
#         #print(df)
#         #print(df.loc[df['Jaccard_top_1'].idxmax()])
#         #df = perf[1]
#         #print(perf[1])
#         #print()
#         #print(perf[1])
#         #print(perf[0])
#         #print(perf[-3])
#         #print(perf[0])
#         # df = perf[3]
#         # print("val jaccard")
#         print(max(perf[-3]))
#         # ind = df['Jaccard_largest_1'].idxmax()
#         # df = perf[1]
#         # print('test jaccard')
#         # print(df.loc[ind]["Jaccard_Largest_1"])
        
        # print("=========")
"""

TE18_sites_dict_4A = torch.load('./../TE18_sites_dict_4A.pt')
RB9_sites_dict_4A = torch.load('./../RB9_sites_dict_4A.pt')
TE18_sites_dict_10A = torch.load('./../TE18_sites_dict_10A.pt')
RB9_sites_dict_10A = torch.load('./../RB9_sites_dict_10A.pt')
ligands = 'ligands'
real_chains_TE18 = {
    "2pwt": ({'pdb_chain':'A'}, {ligands: [("LHA",3)]}), 
    
    "5v3f": ({'pdb_chain':'A'}, {ligands: [("74G",1)]}), 
    
    "379d": ({'pdb_chain':'B'}, {ligands: [("CO",4)]}),
    
    "5bjo": ({'pdb_chain':'E'}, {ligands: [("MG",2), ('747', 1)]}), 
    
    "4pqv": ({'pdb_chain':'A'}, {ligands: [("MG",7)]}), 
    
    "430d": ({'pdb_chain':'A'}, {ligands: [("MG",9)]}), 
    
    "1nem": ({'pdb_chain':'A'}, {ligands: [("BDR",1), ("BDG",1), ("NEB",1), ("IDG",1)]}), 
    
    "1q8n": ({'pdb_chain':'A'}, {ligands: [("MGR",1)]}), 
    
    "1f1t": ({'pdb_chain':'A'}, {ligands: [("ROS",1)]}), 
    
    "2juk": ({'pdb_chain':'A'}, {ligands: [("G0B",1)]}), 
    
    "4yaz": ({'pdb_chain':'R'}, {ligands: [("4BW",1), ("MG", 4)]}), 
    
    "364d": ({'pdb_chain':'C'}, {ligands: [("MG",8)]}),
    
    "6ez0": ({'pdb_chain':'A'}, {ligands: [("U37",5)]}), 
    
    "2tob": ({'pdb_chain':'A'}, {ligands: [("TOA",1), ("2TB",1), ("TOC",1)]}), 
    
    "1ddy": ({'pdb_chain':'A'}, {ligands: [("B12",1), ("NME",1)]}),
    
    "1fmn": ({'pdb_chain':'A'}, {ligands: [("FMN",1)]}), 
    
    "2mis": ({'pdb_chain':'A'}, {ligands: [("MG",4)]}), 
    
    "4f8u": ({'pdb_chain':'B'}, {ligands: [("SIS",2)]}), 
    
    }

TE18_RGCN = []
for data in TE18_pyg:
    TE18_RGCN.append(Data(x=data.x, y=data.y, edge_attr = transform_edge_attr(data.edge_attr), edge_index = data.edge_index))

RB9_RGCN = []
for data in RB9_pyg:
    RB9_RGCN.append(Data(x=data.x, y=data.y, edge_attr = transform_edge_attr(data.edge_attr), edge_index = data.edge_index))
TE18_RGCN_min = []
TE18_pyg_min = torch.load("./../TE18_data_4A_NodeEmbeddings_minimization_removed_nodes.pt")
real_chains_minimized = real_chains_TE18.copy()
del real_chains_minimized['379d']
del real_chains_minimized['364d']


real_chains_TE13 = real_chains_TE18.copy()
del real_chains_TE13['379d']
del real_chains_TE13['364d']
del real_chains_TE13['1ddy']
del real_chains_TE13['6ez0']
del real_chains_TE13['4yaz']

for data in TE18_pyg_min:
    TE18_RGCN_min.append(Data(x=data.x, y=data.y, edge_attr = transform_edge_attr(data.edge_attr), edge_index = data.edge_index))

good_inds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 14, 15]
TE13_RGCN = [x for ind, x in enumerate(TE18_RGCN_min) if ind in good_inds]
good_inds_te18 = [0, 1, 3, 4, 5, 6, 7, 8, 9, 13, 14, 16, 17]
TE13_non_min = [x for ind, x in enumerate(TE18_RGCN) if ind in good_inds_te18]
"""
# test_loader = torch.load('./data/datasets/rnaglib_test_all_6A_64.pt')
# val_loader = torch.load('./data/datasets/rnaglib_val_all_6A_64.pt')

# TE18_loader = torch.load("./data/datasets/rnaglib_TE18_test_4A.pt")
# print(TE18_loader)
# num_iters=3
# plot_counter = 0
# perf_dict_over_pools = {}
# pools = [1, 3, 5, 7, 9]

# for num_layers in [4]:
#     for ind, min_percentile in enumerate([0.9]):
#         perf_dict_over_pools[min_percentile] = {}
#         for min_edge_score in [0.6, 0.75, 0.85]:
#             perf_dict_over_pools[min_percentile][min_edge_score] = {}
#             perf_dict_over_pools[min_percentile][min_edge_score]['Recall'] = []
#             perf_dict_over_pools[min_percentile][min_edge_score]['Precision'] = []
#             perf_dict_over_pools[min_percentile][min_edge_score]['MCC'] = []
#             perf_dict_over_pools[min_percentile][min_edge_score]['AUC'] = []
#             #print(num_layers, min_edge_score)
#             #if num_layers == 3 and min_score > 0.5: continue
#             net = RGCNPoolNet(644, num_layers, min_edge_score, min_percentile)
#             net.load_state_dict(torch.load("./rnaglib_nr_saved_networks/RGCNx"+str(num_layers)+"_edgePool_MSE_"+str(min_edge_score)+"MinEdgeScore_"+str(min_percentile)+"MinPercentile_percent_edge_labels_"+ str(num_iters)+"iters_noOnehop_fixed_DO0.5_WD0.0_ligands_best_val_performance.pt"))
#             MCCs = []
#             for pool_number in pools:
#                 print("Perforamnce statistics", num_layers, min_percentile, min_edge_score)
#                 # print(np.median(res))
#                 # print(TE18_RGCN)
#                 outs = get_x_scores(net, test_loader , pool_number)
#                 outs = torch.cat(outs)
#                 outs = outs.squeeze(1)
#                 targets = [d['graph'].y for d in test_loader]
#                 targets=torch.cat(targets).long()
#                 #print("AUC:", roc_auc_score(targets, outs))
#                 perf_dict_over_pools[min_percentile][min_edge_score]['AUC'].append(roc_auc_score(targets, outs))
#                 outs = torch.round(outs).long()
#                 metric = BinaryConfusionMatrix()
#                 metric.update(outs, targets)
#                 res = metric.compute()
#                 perf_dict_over_pools[min_percentile][min_edge_score]['MCC'].append(matthews_corrcoef(targets, outs))
#                 perf_dict_over_pools[min_percentile][min_edge_score]['Precision'].append(precision_score(targets, outs))
#                 perf_dict_over_pools[min_percentile][min_edge_score]['Recall'].append(recall_score(targets, outs))
#                 #print(res)
#                 #print("MCC:", matthews_corrcoef(targets, outs))
#                 #print("Prev:", precision_score(targets, outs))
#                 #print("Recall:", recall_score(targets, outs))
#                 #print("==============")
# print(perf_dict_over_pools)
# fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 12), gridspec_kw={'hspace': 0.4})
# counter = 0
# for i, k in enumerate(perf_dict_over_pools.keys()):
#     for i2, j in enumerate(perf_dict_over_pools[k].keys()):
#         for i3, l in enumerate(perf_dict_over_pools[k][j].keys()):
#             plot_ind = i3
#             axs[plot_ind//2][plot_ind%2].plot([1, 3, 5, 7, 9], perf_dict_over_pools[k][j][l], linestyle =line_styles[i3], label = str(k)+","+str(j))
#             axs[plot_ind//2][plot_ind%2].legend(fontsize=6)
#             axs[plot_ind//2][plot_ind%2].set_xlabel("number of pooling layers")
#             axs[plot_ind//2][plot_ind%2].set_ylabel(l)
#             axs[plot_ind//2][plot_ind%2].set_title(l+" performance over number of pooling layers")
#             plt.legend()
# plt.show()
#"""