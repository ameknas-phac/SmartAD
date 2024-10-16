import torch
import csv
import numpy as np
import pandas as pd
from torch_geometric.data import Data
# Import MetaFluAD models
from metafluad_models import MetaFluAD

def create_3gram2(query_seq, ref_seq):
    """
    Converts the sequences to sets of 3 amino acids, ex:
    ABCDE -> 'ABC','BCD','CDE'
    """
    HA1 = []
    with open(ref_seq, encoding='utf-8') as f:
        reader = csv.reader(f)
        context1 = [line for line in reader]
        # del context[0]
    f.close()
    with open(query_seq, encoding='utf-8') as f:
        reader = csv.reader(f)
        context2 = [line for line in reader]
        # del context[0]
    f.close()
    context = context1 + context2

    row = []
    for line in (context):
        line = ','.join(line)
        line = line.split(",")
        # print(len(line[2]) - 2)
        for i in range(len(line[1]) - 2):
            HA1.append(line[1][i:i + 3])
        row.append([line[0], HA1])
        HA1 = []
    
    return row

def _3gramtovec2(row, _gram_vec):
    """
    Converts each 3gram to a vector representaion pre learned in _gram_vec
    """
    df = pd.read_csv(_gram_vec, sep = '\t', engine = 'python')
    df = pd.concat([df.loc[:,'words'],df.drop('words', axis = 1).\
                    agg(list,axis = 1)], axis = 1).set_axis(['word','vec'],axis = 1)
    gram_vec = df[["word", "vec"]].set_index("word").to_dict(orient='dict')["vec"]
    
    HA_vec = {}

    for i in range(len(row)):
        HA_temp = []
        test = row[i][1]
        for j in range(len(test)):
            if gram_vec.get(test[j]) == None:
                HA_temp.append(gram_vec.get('<unk>'))
            else:
                HA_temp.append(gram_vec.get(test[j])) 
            if i == 0:
                HA_vec['Reference: '+row[i][0]] = HA_temp
            else:
                HA_vec[row[i][0]] = HA_temp

    return HA_vec
        
def load_x(HA_vec):
    vec = np.array(list(HA_vec.values()))
    vec = torch.FloatTensor(vec)
    return vec.unsqueeze(1)

def seqs_to_geom(query_seq, ref_seq):
    rows = create_3gram2(query_seq, ref_seq)
    # feature_tab = 'W:/Projects/IRVC_Genomics/12-Adam/MetaFluAD Code/data/protVec_100d_3grams.csv'
    feature_tab = 'protVec_100d_3grams.csv'
    HA_vec = _3gramtovec2(rows, feature_tab)
    x = load_x(HA_vec)
    HA_id = list(HA_vec.keys())
    ref_ind = 0
    # Creates an edge from each node to the reference node
    src = torch.Tensor(np.repeat(ref_ind, len(x)-1))
    dst = torch.Tensor(np.array([i for i in range(len(x)) if i != ref_ind]))
    edge_index =torch.stack([src,dst]).to(torch.int64)
    # Return Torch Geometric Data type
    data = Data(x=x, edge_index = edge_index, HA_id=HA_id)

    return data

class metafluad_model():
    def __init__(self):
        # Load model
        self.model = MetaFluAD()
        # pretrained_dict = torch.load('W:/Projects/IRVC_Genomics/12-Adam/MetaFluAD Code/data/result/model_parma/model_H1N1.pth')
        pretrained_dict = torch.load('model_H1N1.pth')
        model_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
    
    def distances(self, query_csv, ref_csv):
        # Predict distance of every sequence to the refernce sequence
        self.model.eval()
        data = seqs_to_geom(query_csv, ref_csv)
        output, feature = self.model(data)
        return torch.exp(output.detach()).numpy()