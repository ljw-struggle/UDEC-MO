import os, torch, numpy as np, pandas as pd
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from sklearn.preprocessing import scale # standardize the data, equivalent to (x - mean(x)) / std(x)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score, roc_auc_score
from sklearn.metrics.cluster import contingency_matrix, rand_score, adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from scipy.optimize import linear_sum_assignment

### Dataset Preparation
class MMDataset(Dataset):
    def __init__(self, data_dir, concat_data=False):
        # Load the multi-omics data and concatenate the modality-specific features.
        self.data_dir = data_dir  # Save data_dir as instance variable
        data_list = []
        modality_list = ['modality_mrna', 'modality_meth', 'modality_mirna'] if data_dir.find('bulk') != -1 else ['modality_rna', 'modality_protein', 'modality_atac']     
        for modality in modality_list:
            modality_data = pd.read_csv(os.path.join(data_dir, modality + '.csv'), header=0, index_col=0) # shape: (num_samples, num_features)
            modality_data_min = np.min(modality_data.values, axis=0, keepdims=True) # shape: (1, num_features)
            modality_data_max = np.max(modality_data.values, axis=0, keepdims=True) # shape: (1, num_features)
            modality_data_values = (modality_data.values - modality_data_min)/(modality_data_max - modality_data_min + 1e-10) # shape: (num_samples, num_features), normalize the data to [0, 1]
            data_list.append(modality_data_values.astype(float))
            print('{} shape: {}'.format(modality, modality_data_values.shape))
        label = modality_data.index.astype(int) # shape: (num_samples, )
        self.categories = np.unique(label).shape[0]; self.data_samples = data_list[0].shape[0]; # number of categories, number of samples
        self.data_views = len(data_list); self.data_features = [data_list[v].shape[1] for v in range(self.data_views)] # number of categories, number of views, number of samples, number of features in each view
        self.concat_data = concat_data
        if self.concat_data:
            self.X = [torch.from_numpy(x).float() for x in data_list]; self.Y = torch.tensor(label, dtype=torch.long)
            self.X = torch.cat(self.X, dim=1) # concatenate the data from different views, shape: (num_samples, sum(num_features))
        else:
            self.X = [torch.from_numpy(x).float() for x in data_list]; self.Y = torch.tensor(label, dtype=torch.long)

    def __getitem__(self, index):
        if self.concat_data:
            x = self.X[index] # select the data from the index
            y = self.Y[index] # select the label from the index
        else:
            x = [x[index] for x in self.X] # convert to tensor
            y = self.Y[index] # select the label from the index
        return x, y, index

    def __len__(self):
        return len(self.Y)
    
    def get_data_info(self):
        return self.data_views, self.data_samples, self.data_features, self.categories
    
    def get_label_to_name(self):
        if self.data_dir == './data/data_sc_multiomics/TEA/':
            return {0: 'B.Activated', 1: 'B.Naive', 2: 'DC.Myeloid', 3: 'Mono.CD14', 4: 'Mono.CD16', 5: 'NK', 6: 'Platelets', 7: 'T.CD4.Memory', 8: 'T.CD4.Naive', 9: 'T.CD8.Effector', 10: 'T.CD8.Naive', 11: 'T.DoubleNegative'}