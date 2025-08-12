
import numpy as np
import pandas as pd
from scipy import io
from RiForest import *
from utils import *
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import time

path_data=r"./dataset/thyroid.mat"
    
data=io.loadmat(path_data)
data_X=pd.DataFrame(data['X'])
data_label=data['y']

scaler = StandardScaler()
data_X = scaler.fit_transform(data_X)

iso = RiForest(sample_size=256, n_trees=100, bins=10, n_iter=5, alpha=0.8)
start = time.time()
iso.fit(data_X)
anomaly_score = iso.anomaly_score(data_X)
end = time.time()

print(roc_auc_score(data_label, anomaly_score))
print(end-start)
    