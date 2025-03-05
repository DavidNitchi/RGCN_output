import torch

from rnaglib.tasks import BindingSite, BenchmarkBindingSite
from rnaglib.transforms import GraphRepresentation, RNAFMTransform
from rnaglib.config import get_modifications_cache
from rnaglib.learning.task_models import PygModel

# Hyperparameters to tune
#loader_batch_size = 64
# ta = BenchmarkBindingSite(root="RNA_Site_Bench", recompute=False, cutoff=4.0)
# ta.dataset.add_representation(GraphRepresentation(framework="pyg"))
# ta.add_feature(RNAFMTransform())
# train_loader, val_loader, test_loader = ta.get_split_loaders(recompute=False)

# torch.save(test_loader, "./data/datasets/rnaglib_TE18_test_4A.pt")

ta = BindingSite(root="bs-all", cutoff=6.0)
train_loader, val_loader, test_loader = ta.get_split_loaders(precomputed=False)
print("Splitting Dataset")
ta.dataset.add_representation(GraphRepresentation(framework="pyg"))
ta.add_feature(RNAFMTransform())
train_loader, val_loader, test_loader = ta.get_split_loaders(recompute=False, batch_size=loader_batch_size)

torch.save(train_loader, "./data/datasets/rnaglib_train_nr_6A_"+str(loader_batch_size)+".pt")
torch.save(val_loader, "./data/datasets/rnaglib_val_nr_6A_"+str(loader_batch_size)+".pt")
torch.save(test_loader, "./data/datasets/rnaglib_test_nr_6A_"+str(loader_batch_size)+".pt")
