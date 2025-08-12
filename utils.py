import numpy as np

def c(size):
    if size > 2:
        return 2 * (np.log(size-1)+0.5772156649) - 2*(size-1)/size
    if size == 2:
        return 1
    return 0

class LeafNode:
    def __init__(self, size, data):
        self.size = size
        self.data = data

class DecisionNode:
    def __init__(self, left, right, splitAtt, splitVal, feattype):
        self.left = left
        self.right = right
        self.splitAtt = splitAtt
        self.splitVal = splitVal
        self.feattype = feattype
        self.path_length = np.nan
