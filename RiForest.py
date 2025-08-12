import numpy as np
import pandas as pd
import random
import math
from utils import *

class RiTree:
    def __init__(self, height, height_limit, bins, alpha, n_iter, _lambda):
        self.height = height
        self.height_limit = height_limit
        self.bins = bins
        self.alpha = alpha
        self.n_iter = n_iter
        self._lambda = _lambda
    
    def _entropy(self, val):
            if val in [0, 1]:
                return 0
            else:
                return val*math.log(val)
            
    def dim_entropy(self, x, dim_bins):
        counts, _ = np.histogram(x, bins=dim_bins)
        probs = counts / counts.sum()
        mask = (probs > 0) & (probs < 1)
        e = -np.sum(probs[mask] * np.log(probs[mask]))
        return e, len(counts)
    
    def entropy_thr(self, dim_bins, alpha):
        ent_max = (-1) * math.log(1/dim_bins)
        return alpha * ent_max
    
    def valley_emp(self, x, bins):
        counts, binedge = np.histogram(x,bins=bins)
        n = np.sum(counts)
        t = list(range(len(counts)+1))
        bin_sum = []
        obj_li = []

        for i in range(len(binedge)-2):
            lf = counts[:i+1]
            lf_v = t[:i+1]
            rt = counts[i+1:]
            rt_v = t[i+1:]
            w1 = np.sum(lf) / n
            w2 = np.sum(rt) / n
            u1 = np.sum([v*(j/np.sum(lf)) for j,v in zip(lf,lf_v)])
            u2 = np.sum([v*(j/np.sum(rt)) for j,v in zip(rt,rt_v)])
            obj_li.append((1-counts[i]/n)*(w1*u1**2+w2*u2**2))
            norm_count = counts/np.sum(counts)
            avg_lf = np.sum(norm_count[:i+1])
            avg_rt = np.sum(norm_count[i+1:])
            bin_sum.append((avg_lf,avg_rt))

        max_idx = np.argmax(obj_li)
        return binedge[max_idx+1], np.array(bin_sum[max_idx])
    
    
    def fit(self, X: np.ndarray):

        if self.height >= self.height_limit or X.shape[0] <= 2:
            self.root = LeafNode(X.shape[0], X)
            return self.root
        
        num_features = X.shape[1]
        non_uni_att = []
        not_same_att = []

        for dim in range(num_features):
            dim_ent, bin_count = self.dim_entropy(X[:,dim],self.bins)
            ent_thr = self.entropy_thr(bin_count, self.alpha)

            if dim_ent <= ent_thr and dim_ent != 0:
                non_uni_att.append(dim)

            if dim_ent != 0:
                not_same_att.append(dim)
        
        for i in range(self.n_iter):    
            randomvec = np.random.uniform(-1,1,size=num_features)
            sparsvec = random.choices([0,np.sqrt(3/(1-self._lambda))],weights = [self._lambda,1-self._lambda],k=num_features)
            normvec = randomvec*sparsvec
            p_X = np.dot(X,normvec)
            dim_ent, bin_count = self.dim_entropy(p_X, self.bins)
            dim_thr = self.entropy_thr(bin_count, self.alpha)
            
            if dim_ent <= dim_thr and dim_ent != 0:
                non_uni_att.append(normvec)
        
        if len(non_uni_att) > 0:
            splitAtt = random.choice(non_uni_att)

            if isinstance(splitAtt, np.ndarray):
                projected_X = np.dot(X,splitAtt)
                splitVal, bin_h = self.valley_emp(projected_X, self.bins)
            else:
                splitVal, bin_h = self.valley_emp(X[:,splitAtt], self.bins)
        else:
            if len(not_same_att) != 0:
                splitAtt = random.choice(not_same_att)
            else:
                splitAtt = np.random.randint(0, num_features)

            splitVal = min(X[:,splitAtt]) + np.ptp(X[:,splitAtt])/2
        
        if isinstance(splitAtt, np.ndarray):
            X_left = X[projected_X < splitVal]
            X_right = X[projected_X >= splitVal]
            feat_type = True
        else:
            X_left = X[X[:, splitAtt] < splitVal]
            X_right = X[X[:, splitAtt] >= splitVal]
            feat_type = False

        left = RiTree(self.height + 1, self.height_limit, self.bins, self.alpha, self.n_iter, self._lambda)
        right = RiTree(self.height + 1, self.height_limit, self.bins, self.alpha, self.n_iter, self._lambda)
        left.fit(X_left)
        right.fit(X_right)
        
        self.root = DecisionNode(left.root, right.root, splitAtt, splitVal, feat_type)
        
        if len(non_uni_att) > 0:
            self.root.path_length = 1 - np.abs(bin_h[0] - bin_h[1])
            
        return self.root

class RiForest:
    def __init__(self, sample_size, bins, alpha, n_iter=10, n_trees=100):
        self.sample_size = sample_size
        self.n_trees = n_trees
        self.bins = bins
        self.alpha = alpha
        self.n_iter = n_iter

    def fit(self, X: np.ndarray):

        self.trees = []
        if isinstance(X, pd.DataFrame):
            X = X.values
        n_rows = X.shape[0]
        height_limit = np.ceil(np.log2(self.sample_size))
        
        for i in range(self.n_trees):
            data_index = np.random.choice(range(n_rows), size=self.sample_size, replace=False)
            X_sub = X[data_index]
            tree = RiTree(0, height_limit, bins = self.bins, alpha = self.alpha, n_iter = self.n_iter, _lambda = random.random())
            tree.fit(X_sub)
            self.trees.append(tree)
        return self

    def path_length(self, X:np.ndarray) -> np.ndarray:
        paths = []

        for row in X:
            path = []

            for tree in self.trees:
                node = tree.root
                length = 0

                while isinstance(node, DecisionNode):
                    if not np.isnan(node.path_length):
                        length += node.path_length
                    else:
                        length += 1
                    
                    if node.feattype:
                        if np.dot(row, node.splitAtt) < node.splitVal:
                            node = node.left
                        else:
                            node = node.right
                    else:
                        if row[node.splitAtt] < node.splitVal:
                            node = node.left
                        else:
                            node = node.right                  
                leaf_size = node.size
                pathLength = length + c(leaf_size)
                path.append(pathLength)
            paths.append(path)
        paths = np.array(paths)
        return np.mean(paths, axis=1)

    def anomaly_score(self, X:pd.DataFrame) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values
        avg_length = self.path_length(X)
        scores = np.array([np.power(2, -l/c(self.sample_size)) for l in avg_length])
        return scores

    def predict(self, X:np.ndarray, threshold:float) -> np.ndarray:
        scores = self.anomaly_score(X)
        prediction = self.predict_from_anomaly_scores(scores, threshold)
        return prediction

