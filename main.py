import torch
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac


torch.manual_seed(0)
# random generate 100 3-way tensor
dataset = torch.rand([100,20,20,20])
factors = parafac(dataset[0], rank=15)


# randomly generate three way tensors
# do the cp decomposition => 3 matrix
# 3 matrix as the network input and output 3 matrix
# 
