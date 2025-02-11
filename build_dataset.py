import os
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.MMCIFParser import FastMMCIFParser
from Bio.PDB.NeighborSearch import NeighborSearch
from Bio.PDB.Selection import unfold_entities
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.DSSP import DSSP
from time import perf_counter
from rnaglib.representations import GraphRepresentation
import numpy as np
import torch
from rnaglib.data_loading import RNADataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
TE18 = ['2pwtA', '5v3fA', '379dB', '5bjoE', '4pqvA', '430dA', '1nemA', '1q8nA', '1f1tA', '2jukA',
            '4yazR', '364dC', '6ez0A', '2tobA', '1ddyA', '1fmnA', '2misA', '4f8uB']
used_pdbs_test_set_4A = torch.load('./../pdbs_ligands_ions_4A_embeddings_min4nbs_0.3seqId_testSet.pt')
used_pdbs_4A = torch.load('./../pdbs_ligands_ions_4A_embeddings_min4nbs_0.3seqId.pt')
TE18_json = [pdb[:-1] for pdb in TE18]

all_pdbs_4A = used_pdbs_4A+used_pdbs_test_set_4A+TE18_json

all_graphs_dataset = RNADataset(redundancy='all', all_graphs = [pdb.lower()+'.json' for pdb in all_pdbs_4A])

IONS = ["3CO", "ACT", "AG", "AL", "ALF", "AU", "AU3", "BA", "BEF", "BO4", "BR", "CA", "CAC", "CD", "CL", "CO",
        "CON", "CS", "CU", "EU3", "F", "FE", "FE2", "FLC", "HG", "IOD", "IR", "IR3", "IRI", "IUM", "K", "LI",
        "LU", "MG", "MLI", "MMC", "MN", "NA", "NCO", "NH4", "NI", "NO3", "OH", "OHX", "OS", "PB", "PO4", "PT",
        "PT4", "RB", "RHD", "RU", "SE4", "SM", "SO4", "SR", "TB", "TL", "VO4", "ZN"]

ALLOWED_ATOMS = ['C', 'H', 'N', 'O', 'Br', 'Cl', 'F', 'P', 'Si', 'B', 'Se']
ALLOWED_ATOMS += [atom_name.upper() for atom_name in ALLOWED_ATOMS]
ALLOWED_ATOMS = set(ALLOWED_ATOMS)
edge_map={'B35': 0, 'B53': 1, 'CHH': 2, 'CHS': 3, 'CHW': 4, 'CSH': 5, 'CSS': 6, 'CSW': 7, 'CWH': 8, 'CWS': 9, 'CWW': 10, 'THH': 11, 'THS': 12, 'THW': 13, 'TSH': 14, 'TSS': 15, 'TSW': 16, 'TWH': 17, 'TWS': 18, 'TWW': 19}


def is_dna(res):
    """
    Returns true if the input residue is a DNA molecule

    :param res: biopython residue object
    """
    if res.id[0] != ' ':
        return False
    if is_aa(res):
        return False
    # resnames of DNA are DA, DC, DG, DT
    if 'D' in res.get_resname():
        return True
    else:
        return False


def hariboss_filter(lig, cif_dict, mass_lower_limit=160, mass_upper_limit=1000):
    """
    Sorts ligands into ion / ligand / None
     Returns ions for a specific list of ions, ligands if the hetatm has the right atoms and mass and None otherwise

    :param lig: A biopython ligand residue object
    :param cif_dict: The output of the biopython MMCIF2DICT object
    :param mass_lower_limit:
    :param mass_upper_limit:

    """
    lig_name = lig.id[0][2:]
    if lig_name == 'HOH':
        return None

    if lig_name in IONS:
        return 'ion'
        #this will prevent ions being added to the dataset
        #return None

    lig_mass = float(cif_dict['_chem_comp.formula_weight'][cif_dict['_chem_comp.id'].index(lig_name)])
    #SPM is a known artifact specifically filtering it out
    if lig_name == 'SPM':
        return None
    if lig_mass < mass_lower_limit or lig_mass > mass_upper_limit:
        return None
    ligand_atoms = set([atom.element for atom in lig.get_atoms()])
    if 'C' not in ligand_atoms:
        return None
    if any([atom not in ALLOWED_ATOMS for atom in ligand_atoms]):
        return None
    return 'ligand'


def get_mmcif_graph_level(mmcif_dict):
    """
    Parse an mmCIF dict and return some metadata.

    :param cif: output of the Biopython MMCIF2Dict function
    :return: dictionary of mmcif metadata (for now only resolution terms)
    """
    keys = {'resolution_low': '_reflns.d_resolution_low',
            'resolution_high': '_reflns.d_resolution_high',
            'pdbid': '_pdbx_database_status.entry_id'
            }

    annots = {}
    for name, key in keys.items():
        try:
            annots[name] = mmcif_dict[key]
        except KeyError:
            pass
    return annots


# def get_hetatm(cif_dict):
#     all_hetatm = set(cif_dict.get('_pdbx_nonpoly_scheme.mon_id', []))
#     return all_hetatm

def get_small_partners(cif, mmcif_dict=None, radius=4, mass_lower_limit=160, mass_upper_limit=1000):
    """
    Returns all the relevant small partners in the form of a dict of list of dicts:
    {'ligands': [
                    {'id': ('H_ARG', 47, ' '),
                     'name': 'ARG'
                     'rna_neighs': ['1aju.A.21', '1aju.A.22', ... '1aju.A.41']},
                  ],
     'ions': [
                    {'id': ('H_ZN', 56, ' '),
                     'name': 'ZN',
                     'rna_neighs': ['x', y , z]}
                     }

    :param cif: path to a mmcif file
    :param mmcif_dict: if it got computed already
    :return:
    """
    structure_id = cif[-8:-4]
    # print(f'Parsing structure {structure_id}...')

    mmcif_dict = MMCIF2Dict(cif) if mmcif_dict is None else mmcif_dict
    parser = FastMMCIFParser(QUIET=True)
    structure = parser.get_structure(structure_id, cif)

    atom_list = unfold_entities(structure, 'A')
    neighbors = NeighborSearch(atom_list)

    all_interactions = {'ligands': [], 'ions': []}

    model = structure[0]
    for res_1 in model.get_residues():
        # Only look around het_flag
        het_flag = res_1.id[0]
        if 'H' in het_flag:
            # hariboss select the right heteroatoms and look around ions and ligands
            selected = hariboss_filter(res_1, mmcif_dict,
                                       mass_lower_limit=mass_lower_limit,
                                       mass_upper_limit=mass_upper_limit)
            if selected is not None:  # ion or ligand
                interaction_dict = {'id': res_1.id, 'name': res_1.id[0][2:]}
                found_rna_neighbors = set()
                for atom in res_1:
                    # print(atom)
                    for res_2 in neighbors.search(atom.get_coord(), radius=radius, level='R'):
                        # Select for interactions with RNA
                        if not (is_aa(res_2) or is_dna(res_2) or 'H' in res_2.id[0]):
                            # We found a hit
                            rglib_resname = '.'.join([structure_id, str(res_2.get_parent().id), str(res_2.id[1])])
                            found_rna_neighbors.add(rglib_resname)
                if len(found_rna_neighbors) > 0:
                    found_rna_neighbors = sorted(list(found_rna_neighbors))
                    #print(found_rna_neighbors)
                    interaction_dict['rna_neighs'] = found_rna_neighbors
                    all_interactions[f"{selected}s"].append(interaction_dict)
    return all_interactions


def add_graph_annotations(g, cif):
    """
    Adds information at the graph level and on the small molecules partner of an RNA molecule

    :param g: the nx graph created from dssr output
    :param cif: the path to a .mmcif file
    :return: the annotated graph, actually the graph is mutated in place
    """
    mmcif_dict = MMCIF2Dict(cif)
    # Add graph level like resolution
    graph_level_annots = get_mmcif_graph_level(mmcif_dict=mmcif_dict)
    g.graph.update(graph_level_annots)

    # Fetch interactions with small molecules and ions
    all_interactions = get_small_partners(cif, mmcif_dict=mmcif_dict)
    g.graph.update(all_interactions)

    # First fill relevant nodes
    for interaction_dict in all_interactions['ligands']:
        ligand_id = interaction_dict['id']
        for rna_neigh in interaction_dict['rna_neighs']:
            # In some rare cases, dssr removes a residue from the cif, in which case it can be fou
            # in the interaction dict but not in graph...
            if rna_neigh in g.nodes:
                g.nodes[rna_neigh]['binding_small-molecule'] = ligand_id
    for interaction_dict in all_interactions['ions']:
        ion_id = interaction_dict['id']
        for rna_neigh in interaction_dict['rna_neighs']:
            # In some rare cases, dssr removes a residue from the cif, in which case it can be fou
            # in the interaction dict but not in graph...
            if rna_neigh in g.nodes:
                g.nodes[rna_neigh]['binding_ion'] = ion_id
    # Then add a None field in all other nodes
    for node, node_data in g.nodes(data=True):
        if 'binding_ion' not in node_data:
            node_data['binding_ion'] = None
        if 'binding_small-molecule' not in node_data:
            node_data['binding_small-molecule'] = None
    return g


def annotate_proteinSSE(g, structure, pdb_file):
    """
    Annotate protein_binding node attributes with the relative SSE
    if available from DSSP

    :param g: (nx graph)
    :param structure: (PDB structure)

    :return g: (nx graph)
    """

    model = structure[0]
    tic = perf_counter()
    dssp = DSSP(model, pdb_file, dssp='mkdssp', file_type='DSSP')
    toc = perf_counter()

    print(dssp.keys())

    a_key = list(dssp.keys())[2]

    print(dssp[a_key])

    print(f'runtime = {tic - toc:0.7f} seconds')

    return g

def get_binding_sites(all_pdbs):
    binding_site_dicts = {}
    i = 0
    for pdb in all_pdbs:
        i += 1
        if i %100 == 99:
            print(i)
        path = './pdb_data/'+pdb+'.cif'
        if not os.path.isfile('./pdb_data/'+pdb+'.cif'):
            print(pdb)
            continue
        res = get_small_partners(path)
        binding_site_dicts[pdb] = res
        #break
    return binding_site_dicts

def get_all_neighbors_pdb_ligand(sites_dict_pdb):
    s = set()
    for lig in sites_dict_pdb['ligands']:
        #could add an if statement to only select certain sizes of neighbours
        if len(lig['rna_neighs']) < 4:
            continue
        for n in lig['rna_neighs']:
            s.add(n)
    if len(s) == 0:
        return -1
    return s

def get_all_neighbors_pdb_ion(sites_dict_pdb):
    s = set()
    for ion in sites_dict_pdb['ions']:
        if len(ion['rna_neighs']) < 4:
            continue
        for n in ion['rna_neighs']:
            s.add(n)
    if len(s) == 0:
        return -1
    return s

def get_all_neighbors_pdb_both(sites_dict_pdb):
    s1 = get_all_neighbors_pdb_ligand(sites_dict_pdb)
    s2 = get_all_neighbors_pdb_ion(sites_dict_pdb)
    if s1 == -1 and s2 == -1:
        return -1
    elif s1 == -1:
        return s2
    elif s2 == -1:
        return s1
    else:
        s = s1|s2
        return s
    
def make_feature_dict_from_nx(graph, sites_dict_pdb):
    nt_encoding = {'A': [1, 0, 0, 0], 'U': [0, 1, 0, 0], 'G':[0, 0, 1, 0], 'C':[0, 0, 0, 1]}
    features_dict = {'nt_targets': {}, 'nt_features': {}}
    chains_dict = {}
    all_neighs = get_all_neighbors_pdb_ligand(sites_dict_pdb)
    if all_neighs == -1:
        return -1
    '''
    one_hop_nodes = set()
    for neigh in all_neighs:
        #print(graph.nodes())
        one_hop = graph.neighbors(neigh[:4].lower()+neigh[4:])
        for hop_n in one_hop:
            one_hop_nodes.add(hop_n.upper())
    all_neighs = all_neighs | one_hop_nodes
    '''
    #print(all_neighs)
    #print(len(g.edges()))
    for n in sorted(graph.nodes()):
        pdb, chain, nt = n.split('.')
        #nt_encoded = nt_encoding[graph.nodes[n]['nt_code'].upper()]
        #features_dict['nt_features'][n] = torch.FloatTensor(nt_encoded)
        
        #THIS IS THE CODE TO USE IF YOU WANT THE DATASET WITH RNAFM EMBEDDINGS
        key = pdb+'.'+chain
        if key not in chains_dict:
            chains_dict[key] = np.load('./rfam-embeddings/rnafm-embeddings/'+key+'.npz')
        
        node_embedding = chains_dict[key][n]
        features_dict['nt_features'][n] = torch.FloatTensor(node_embedding)
        #print(n)
        in_pocket = 1 if n.upper() in all_neighs else 0
        features_dict['nt_targets'][n] = in_pocket
    #print(features_dict['nt_targets'])
    return features_dict

def build_edge_encoding(edge_map):
    edge_encoding = {}
    for k in edge_map:
        arr = np.zeros(20)
        arr[edge_map[k]] = 1
        edge_encoding[k] = torch.FloatTensor(arr)
    return edge_encoding

def make_pyg_data_from_nx(graph, features_dict, edge_map):
    edge_encoding = build_edge_encoding(edge_map)
    node_map = {n: i for i, n in enumerate(sorted(graph.nodes()))}
    x = torch.stack([features_dict['nt_features'][n] for n in
                         sorted(graph.nodes())]) if 'nt_features' in features_dict else None
    y = [features_dict['nt_targets'][n] for n in sorted(graph.nodes())] if 'nt_targets' in features_dict else None
    y = torch.FloatTensor(y)
    #print(y)
    #y = torch.stack(
        #[features_dict['nt_targets'][n] for n in sorted(graph.nodes())]) if 'nt_targets' in features_dict else None
    edge_index = [[node_map[u], node_map[v]] for u, v, data in sorted(graph.edges(data=True), key=lambda element: (element[0], element[1]))]
    edge_index = torch.tensor(edge_index, dtype=torch.long).T
    #np.zeros(18)
    edge_attrs = [edge_encoding[data['LW'].upper()] for u, v, data in sorted(graph.edges(data=True), key=lambda element: (element[0], element[1]))]
    #print(edge_attrs)
    edge_attrs = [e.unsqueeze(0) for e in edge_attrs]
    edge_attrs = torch.cat(edge_attrs)
    #print(edge_attrs.shape)
    return Data(x=x, y=y, edge_attr=edge_attrs, edge_index=edge_index)

def make_data_obj_from_nx(graph, sites_dict_pdb, edge_map):
    
    features_dict = make_feature_dict_from_nx(graph, sites_dict_pdb)
    #print(features_dict['nt_targets'])
    if features_dict ==-1:
        return -1
    data = make_pyg_data_from_nx(graph, features_dict, edge_map)
    return data

def build_pyg_dataset(sites_dict, edge_map, test_pdbs):
    """ 
    build a dataset for the model
    sites_dict is a binding site dictionary which is generated from the get_binding_sites function
    edge_map is a dictionary mapping each name of an edge to a number
    test_pdbs is a set of pdbs not to include from the sites dictionary
    """
    all_data = []
    missing_embeddings = []
    used_pdbs = []
    for pdb in sites_dict:
        print(pdb)
        if pdb not in test_pdbs:
            continue
        try:
            glib_data = all_graphs_dataset.get_pdbid(pdb)
        except:
            print("not found in RNADataset")
            continue
        g = glib_data['rna']
        try:
            data = make_data_obj_from_nx(g, sites_dict[pdb], edge_map)
        except:
            #print('weird nt code')
            #continue
            print('embedding missing for:', pdb)
            missing_embeddings.append(pdb)
            continue
        if data != -1:  
            all_data.append(data)
            used_pdbs.append(pdb)
    return all_data, missing_embeddings, used_pdbs

"""
USAGE: 
first make a binding sites dictionary by calling get binding sites with the list of pdbs
then just call build_pyg_dataset with the edgemap encoding the sites dictionary and t
he list of pdbs form the sites dictionary that should not be included in the dataset.
"""