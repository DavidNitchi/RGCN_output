import os
import torch

dloader = torch.load('./data/datasets/rnaglib_train_nr_64.pt')
counter = 0
for d in dloader:
    counter+=1
print(counter)

# sites_dict_rnaglib = {}
# pdbs = []
# train_pdbs = []
# test_pdbs = []
# val_pdbs = []
# with open("./bs-nr/train_idx.txt", "r") as file:
#     train_inds = [line.strip() for line in file.readlines()]
# with open("./bs-nr/test_idx.txt", "r") as file:
#     test_inds = [line.strip() for line in file.readlines()]
# with open("./bs-nr/val_idx.txt", "r") as file:
#      val_inds = [line.strip() for line in file.readlines()]

# directory = "your_directory_path"
# files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
