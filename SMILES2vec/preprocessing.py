import numpy as np
import pandas as pd
import torch

def smiles_preprocess(smiles):
    ''' replace some notations, does not add EOM '''
    smiles = smiles.replace('Cl','l')
    smiles = smiles.replace('Br','r')
    return smiles

def smiles_mapping(smiles_list, threshold=30):
    ''' return the mapping (including EOM) and maxlength for a list of smiles '''
    smiles_list = [smiles_preprocess(i) for i in smiles_list]
    total_str = ''.join(smiles_list)
    common_char = set(total_str)
    uncommon_char = set()
    temp = common_char.copy()
    for i in temp:
        if(total_str.count(i)<threshold):
            common_char.remove(i)
            uncommon_char.add(i)
    mapping = {}
    for (num,char) in enumerate(common_char):
        mapping[char]=num
    UNCOMMON = max(mapping.values()) + 1
    for char in uncommon_char:
        mapping[char]=UNCOMMON
    EOM = max(mapping.values()) + 1
    mapping['?'] = EOM
    maxlength = max([len(i) for i in smiles_list])
    return mapping, maxlength

def smiles2onehot(smiles,mapping,maxlength):
    ''' return one-hot encoding of length ::maxlegnth:: according to a ::mapping:: '''
    # replace multilength element as one char, add EOM at the end
    smiles = smiles_preprocess(smiles)
    # raise error if maxlength is less than input SMILES
    if (maxlength<len(smiles)):
        print(smiles)
        raise(AssertionError("max length %d less than length of input SMILES string %d" % (maxlength, len(smiles)) ))
    smiles += '?'
    onehot_size = max(mapping.values()) + 1
    onehot = np.zeros([len(smiles), onehot_size]) #plus one for EOM
    longEncoding = np.zeros([len(smiles), 1])
    for (row,char) in enumerate(smiles):
        onehot[row][mapping[char]] = 1
        longEncoding[row] = mapping[char]
    return onehot, longEncoding

def demapping(mapping):
    demapping = {} 
    for key, val in mapping.items():
        demapping[val] = key
    for each in set(mapping.values()):
        if list(mapping.values()).count(each) > 1:
            demapping[each] = '_'
            break
    return demapping

def preprocessing():
    freesolv_data = pd.read_json(r"data/FreeSolv_database.json").T
    fs_smiles = list(freesolv_data['smiles'])
    mapping, maxlength = smiles_mapping(fs_smiles)
    X_raw = [torch.LongTensor(smiles2onehot(i, mapping, maxlength)[1]) for i in fs_smiles]
    X = [torch.FloatTensor(smiles2onehot(i, mapping, maxlength)[0]) for i in fs_smiles]
    y = freesolv_data['calc_s (cal/mol.K)'].copy()
    y[y.isnull()] = freesolv_data[y.isnull()]['calc_s']
    y = np.array(y,dtype=np.float)
    y = (y - y.mean())/ y.std()
    y = torch.tensor(y)
    return X, X_raw, y, demapping(mapping)
