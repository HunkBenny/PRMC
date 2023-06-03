import numpy as np
from typing import Tuple
def ranking_based_evaluation(true,pred,renormalize=False)->Tuple[float,float]:
    """Ranking based evaluation. Uses the methodology as described in https://www.researchgate.net/publication/220283688


    Args:
        pred (float): pred
        true (float): true
        renormalize (bool, optional): renormalize to put in the interval 0 and 1. Defaults to False.

    Returns:
        Tuple[float,float]: T and R
    """

    idx = pred > 0 #this will also incorporate the first moment at which the true rul equals zero :)
    pred=pred[idx]
    true=true[idx]

    #negate to sort in desc order
    true = true[np.argsort(-pred)]
    true = np.argsort(-true)

    T=0
    R=0
    n = true.shape[0]
    for j,sj in enumerate(true):
        #j = iter
        #sj = rank at position j
        if iter == 0:
            continue #because in the first iter there are no ranks to compare to :)
        T+=(true[:j]>sj).sum()
        R+=sum(j-(np.where(true[:j]>sj))[0])
    #normalize:
    T = 1-((4*T)/(n*(n-1)))
    R = 1-((12*R)/(n*(n-1)*(n+1)))

    if renormalize:
        T = 0.5+T/2
        R = 0.5+R/2

    return T,R