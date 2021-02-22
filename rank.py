import numpy as np
import cp
import torch


def random3WayTensor(cpRank):
    # generate 100 10*10*10 tensors with specified cp rank
    tensorset = np.random.rand(1000, 10, 10, 10)
    tensorWithSpecifiedCpRank = []
    for i in tensorset:
        normalizeFactor, factor, _ = cp.cp_als(i, cpRank)
        predictedTensor = cp.cp_compose(normalizeFactor, factor, cpRank)
        tensorWithSpecifiedCpRank.append(predictedTensor)
    tensorWithSpecifiedCpRank = np.array(tensorWithSpecifiedCpRank)
    print(tensorWithSpecifiedCpRank.shape)
    assert tensorWithSpecifiedCpRank.shape[0] == 1000
    return tensorWithSpecifiedCpRank


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    # dataset10 = random3WayTensor(10)
    # dataset50 = random3WayTensor(50)
    dataset100 = random3WayTensor(100)
    np.save('rank100', dataset100)
