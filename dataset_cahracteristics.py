import torch
import pickle
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from RGCNPool import transform_edge_attr
import random
import matplotlib.pyplot as plt
all_ligands_data_4A = torch.load('./../g_data_ligands_4A_embeddings_min4nbs_0.3seqId.pt')
#all_ligands_data_onehop_4A = torch.load('./../g_data_ligands_4A_onehop_embeddings_min4nbs_0.3seqId.pt')
#used_pdbs_ligands_onehop_4A = torch.load('./../pdbs_ligands_4A_onehop_embeddings_min4nbs_0.3seqId.pt')
used_pdbs_ligands_4A = torch.load('./../pdbs_ligands_4A_embeddings_min4nbs_0.3seqId.pt')
sites_dict_4A = pickle.load(open('./../Hariboss+PDBbind+RCSB_data_4A_recent.pkl', 'rb'))
test_set_ligands_4A = torch.load('./../g_data_ligands_4A_embeddings_min4nbs_0.3seqId_testSet_fixed.pt')
used_pdbs_ligands_test_set_4A = torch.load('./../pdbs_ligands_4A_embeddings_min4nbs_0.3seqId_testSet_fixed.pt')
TE18_sites_dict_4A = torch.load('./../TE18_sites_dict_4A.pt')

print(sites_dict_4A.keys())

# ind = used_pdbs_ligands_test_set_4A.index('1FMN')
# print(ind)
# print(test_set_ligands_4A[ind])
# #print(len(used_pdbs_ligands_4A))
# #print(used_pdbs_ligands_test_set_4A)

# val_data_percent = 0.15
# #print(-int(len(RGCN_data)*val_data_percent))
# val_data = all_ligands_data_4A[-int(len(all_ligands_data_4A)*val_data_percent):]
# val_pdbs = used_pdbs_ligands_4A[-int(len(all_ligands_data_4A)*val_data_percent):]
# train_data = all_ligands_data_4A[:-int(len(all_ligands_data_4A)*val_data_percent)]
# train_pdbs = used_pdbs_ligands_4A[:-int(len(all_ligands_data_4A)*val_data_percent)]

# val_inds = random.sample(range(0, len(train_data)), int(len(all_ligands_data_4A)*val_data_percent))
# #print(val_inds)
# val_inds = [89, 73, 96, 149, 238, 110, 221, 119, 102, 92, 74, 286, 285, 174, 177, 155, 127, 226, 211, 19, 282, 172, 71, 93, 291, 235, 67, 258, 208, 261, 68, 50, 239, 252, 18, 69, 278, 4, 231, 86, 237, 233, 62, 294, 140, 251, 160, 269, 205, 270, 112, 43]
# positives = 0
# total = 0
# for d in all_ligands_data_4A:
#     total += d.y.shape[0]
#     positives += sum(d.y)
# print(total)
# print(positives)
# print(len(all_ligands_data_4A))

# rnaglib_test = torch.load("./data/datasets/rnaglib_train_nr.pt")
# for batch in rnaglib_test:
#     g = batch["graph"]
#     print(g)
#     break
#torch.save(val_inds, "./validation_indices.pt")
#for pdb in used_pdbs_ligands_test_set_4A:
    #occ = sites_dict_4A[pdb]
    #for site in occ['ligands']:

#print(sites_dict_4A[used_pdbs_ligands_4A[0]])
# sizes = []
# for pdb in used_pdbs_ligands_4A:
#     for lig in sites_dict_4A[pdb]['ligands']:
#         size = len(lig['rna_neighs'])  
#         if size <4: continue
#         sizes.append(len(lig['rna_neighs']))
# plt.xticks(range(0, 21))
# plt.xlabel("number of residues in in the pocket")

# plt.title("Distribution of pocket sizes in training data")
# plt.hist(sizes)
# plt.show()
# sizes = []
# for pdb in used_pdbs_ligands_test_set_4A:
#     for lig in sites_dict_4A[pdb]['ligands']:
#         size = len(lig['rna_neighs'])  
#         if size <4: continue
#         sizes.append(len(lig['rna_neighs']))
# plt.xticks(range(0, 21))
# plt.xlabel("number of residues in in the pocket")
# plt.title("Distribution of pocket sizes in test data")
# plt.hist(sizes)
# plt.show()
# sizes = []
# for pdb in TE18_sites_dict_4A:
#     for lig in TE18_sites_dict_4A[pdb]['ligands']:
#         size = len(lig['rna_neighs'])  
#         #if size <4: continue
#         sizes.append(len(lig['rna_neighs']))
# plt.xticks(range(0, 21))
# plt.xlabel("number of residues in in the pocket")

# plt.title("Distribution of pocket sizes in TE18")
# plt.hist(sizes)
# plt.show()
"""
def get_sizes(dataset):
    sizes = []
    for d in dataset:
        sizes.append(d.y.shape[0])
    return sizes

print(np.median(get_sizes(train_data)))
print(np.median(get_sizes(val_data)))
print(np.median(get_sizes(test_set_ligands_4A)))


RGCN_data = []
for data in all_ligands_data_4A:
    RGCN_data.append(Data(x=data.x, y=data.y, edge_attr = transform_edge_attr(data.edge_attr), edge_index = data.edge_index))

test_4A_RGCN = []
for data in test_set_ligands_4A:
    test_4A_RGCN.append(Data(x=data.x, y=data.y, edge_attr = transform_edge_attr(data.edge_attr), edge_index = data.edge_index))

val_data_percent = 0.15
#print(-int(len(RGCN_data)*val_data_percent))
val_RGCN = RGCN_data[-int(len(RGCN_data)*val_data_percent):]
val_pdbs = used_pdbs_ligands_4A[-int(len(RGCN_data)*val_data_percent):]
train_RGCN = RGCN_data[:-int(len(RGCN_data)*val_data_percent)]
train_pdbs = used_pdbs_ligands_4A[:-int(len(RGCN_data)*val_data_percent)]

#print(val_data[1].edge_attr)

def get_all_edges(dataset):
    tmp = torch.tensor([])
    print(len(dataset))
    counter = 0
    for d in dataset:
        print(counter)
        counter += 1
        tmp = torch.cat((tmp, d.edge_attr))
    return tmp.tolist()

import matplotlib.pyplot as plt
from collections import Counter

# Sample data: list of edge types
edge_types = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

# Count occurrences of each edge type
edge_counts = Counter(edge_types)

for dataset in [train_RGCN, val_RGCN, test_4A_RGCN]:

    # Ensure all 20 edge types are represented, even if their count is zero
    all_edge_types = get_all_edges(dataset)
    #print(all_edge_types[:10])
    #print(all_edge_types[:10].count(0))
    counts = [all_edge_types.count(e)/len(dataset) if e not in [0, 1, 10] else 0 for e in range(0, 20)]
    print(counts)
    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.bar(edge_types, counts, color='skyblue', edgecolor='black')
    plt.xlabel('Edge Type')
    plt.ylabel('Count')
    plt.title('Number of Occurrences of Each Edge Type')
    plt.xticks(edge_types)  # Ensure all edge types are labeled
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show the plot
    plt.tight_layout()
    plt.show()
"""