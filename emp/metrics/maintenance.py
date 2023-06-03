import warnings
from collections.abc import Iterable
from functools import lru_cache
from typing import Union
import tensorflow as tf
import numpy as np
import scipy.stats as st
import scipy.integrate as integrate
from functools import partial

def _calculate_PRMC(pred:np.ndarray, true:np.ndarray, tau:int, threshold:int, cost_reactive:float, cost_predictive:float, cost_rul:Union[float,np.ndarray])->np.ndarray:
    """
    calculate prmc for multiple machines at multiple points in time

    Args:
        pred (np.ndarray): preds
        true (np.ndarray): trues
        tau (int): lead times
        threshold (int): threshold
        cost_reactive (float): cost reactive
        cost_predictive (float): cost predictive
        cost_rul (Union[float,np.ndarray]): cost of rul

    Returns:
        np.ndarray: prmc
    """

    coords = np.argwhere(pred <= threshold)
    machs = set(coords.T[0])  # machines that are predicted to fail
    # init with inf! because what if the pred is never < than threshold
    pred_pos = np.repeat(np.inf, (pred.shape[0]*2)).reshape((pred.shape[0], 2))
    for mach in machs:
        pred_pos[mach] = (
            (mach, np.amin(coords[coords.T[0] == mach].T[1]) + tau))  # add lead time

    # perceived failures per machine
    true_pos = np.array([(x, y) for x, y in enumerate(true.argmin(axis=1))])

    lost_rul = true_pos[:, 1]-pred_pos[:, 1]

    # two dimensional matrices: containing the coords of the failures or preventions
    # col 0 contains x coord and col 1 contains y coord
    failures = true_pos[true_pos[:, 1] < pred_pos[:, 1]]
    prevented = (pred_pos[true_pos[:, 1] >= pred_pos[:, 1]]).astype(np.int32)

    cost = np.zeros(pred.shape)
    if len(failures) != 0:
        for mach in failures:
            cost[tuple(mach.astype(int))] = cost_reactive

    if len(prevented) != 0:
        if (type(cost_rul) == np.ndarray) or (type(cost_rul) == tf.TensorArray):
            # TODO: does this work for TensorArray?
            # if array, only look at the machines that
            for mach in prevented:
                cost[(mach[0], mach[1])] = cost_predictive + \
                    cost_rul[mach[0]] * lost_rul[mach[0]]
        else:
            for mach in prevented:
                cost[tuple(mach.astype(int))] = cost_predictive + \
                    cost_rul * lost_rul[int(mach[0])]

    return cost

def _calculate_PRMC_one_dim(pred_rul:np.ndarray, true_rul:np.ndarray, tau:int, threshold:int, cost_reactive:float, cost_predictive:float, cost_rul:Union[float,np.ndarray])->np.ndarray:
    """
    calculate prmc for multiple machines at the same point in time

    Args:
        pred_rul (np.ndarray): preds
        true_rul (np.ndarray): true
        tau (int): leadtime
        threshold (int): threshold
        cost_reactive (float): reactive cost
        cost_predictive (float): predictive cost
        cost_rul (Union[float,np.ndarray]): cost of rul

    Returns:
        np.ndarray: prmc
    """
    cost = np.zeros(len(pred_rul))

    # schedule maintenance (PdM)
    schedule_maintenance = np.where(pred_rul <= threshold, True, False)
    if (type(cost_rul) == np.ndarray) or (type(cost_rul) == tf.TensorArray):
        cost[schedule_maintenance] = (
            true_rul[schedule_maintenance] - tau)*cost_rul[schedule_maintenance]+cost_predictive
    else:
        cost[schedule_maintenance] = (
            true_rul[schedule_maintenance] - tau)*cost_rul+cost_predictive

    # will break (Reactive maintenance)
    will_break = np.where(true_rul < tau, True, False)
    cost[will_break] = cost_reactive

    return cost

def _calculate_PRMC_point(pred_rul:float, true_rul:float, tau:int, threshold:int, cost_reactive:float, cost_predictive:float, cost_rul:float)->float:
    """
    calculate PRMC at one point in time for one machine

    Args:
        pred_rul (float): preds
        true_rul (float): true
        tau (int): lead times
        threshold (int): thresholds
        cost_reactive (float): cost reactive
        cost_predictive (float): cost predictive
        cost_rul (float): cost rul. This can be a float (shared costs) or an array with the same number of rows as the number of machines and one column

    Returns:
        float: prmc
    """
    if true_rul < tau:
        return cost_reactive

    if pred_rul <= threshold:
        return cost_predictive + (true_rul - tau)*cost_rul

    return 0

def calculate_PRMC(pred:Union[float,np.ndarray], true:Union[float,np.ndarray], tau:int, threshold:int, cost_reactive:float, cost_predictive:float, cost_rul:Union[float,np.ndarray])->Union[float,np.ndarray]:
    """
    Calculate PRMC, this method looks at the format of its inputs and passes the inputs to the correct private method.

    USAGE:
    If the PRMC needs to be calculated for one machine, at one point in time
    If the PRMC needs to be calculated for multiple machines, at the same point in time (will linearly forecast the predictions)
    If the PRMC needs to be calculated for multiple machines, at multiple points in time (i.e. the simulation process as explained in Section 6 in the dissertation)

    Args:
        pred (Union[float,np.ndarray]): predictions
        true (Union[float,np.ndarray]): true values
        tau (int): lead time
        threshold (int): threshold
        cost_reactive (float): cost of reactive maintenance
        cost_predictive (float): cost of predictive maintenance
        cost_rul (Union[float,np.ndarray]): cost of ruls

    Returns:
        Union[float,np.ndarray]: calculated PRMC 
    """

    if type(true) == np.ndarray:
        if true.ndim == 1:
            return _calculate_PRMC_one_dim(pred, true, tau, threshold, cost_reactive, cost_predictive, cost_rul)
        else:
            return _calculate_PRMC(pred, true, tau, threshold, cost_reactive, cost_predictive, cost_rul)

    return _calculate_PRMC_point(pred, true, tau, threshold, cost_reactive, cost_predictive, cost_rul)

def calculate_EPRMC(pred_rul:Union[float,np.ndarray], true_rul:Union[float,np.ndarray], threshold:int, cost_reactive:float, cost_predictive:float, cost_rul:Union[float,np.ndarray], limit:int=150, tau_distr=partial(st.lognorm.pdf,s=1.2, scale=6))->Union[float,np.ndarray]:
    """
    Calculates EPRMC.
    If no tau_distr is not specified, the expected value of the default distribution equals approx. 12 time units. 

    USAGE:
    If the EPRMC needs to be calculated for one machine, at one point in time
    If the EPRMC needs to be calculated for multiple machines, at the same point in time (will linearly forecast the predictions)
    If the EPRMC needs to be calculated for multiple machines, at multiple points in time (i.e. the simulation process as explained in Section 6 in the dissertation)

    Args:
        pred_rul (Union[int,np.ndarray]): preds
        true_rul (Union[int,np.ndarray]): true
        threshold (int): threshold
        cost_reactive (float): cost reactive
        cost_predictive (float): cost predictive
        cost_rul (Union[float,np.ndarray]): cost rul
        limit (int, optional): limit for integration. Defaults to 150.
        tau_distr (functools.partial, optional): distribution of lead times. Defaults to partial(st.lognorm.pdf,s=1.2, scale=6).

    Returns:
        Union[float,np.ndarray]: EPRMC
    """
    if type(true_rul) == np.ndarray:
        if true_rul.ndim == 1:
            return _calculate_EPRMC_one_dim(pred_rul, true_rul, tau_distr, threshold, cost_reactive, cost_predictive, cost_rul, limit=limit)
        else:
            return _calculate_EPRMC_two_dim(pred_rul, true_rul, tau_distr, threshold, cost_reactive, cost_predictive, cost_rul, limit=limit)
    else:
        return _calculate_EPRMC(pred_rul, true_rul, tau_distr, threshold, cost_reactive, cost_predictive, cost_rul, limit=limit)
    
def _calculate_EPRMC(pred_rul:float, true_rul:float, tau_distr, threshold:int, cost_reactive:float, cost_predictive:float, cost_rul:float, limit:int=150)->float:
    """
    calculate eprmc for one machine at one point in time

    Args:
        pred_rul (float): pred
        true_rul (float): true
        tau_distr (functools.partial): lead time
        threshold (int): threshold
        cost_reactive (float): cost reactive
        cost_predictive (float): cost predictive
        cost_rul (float): cost rul
        limit (int, optional): limit for integration. Defaults to 150.

    Returns:
        float: EPRMC
    """
    return integrate.quad(lambda tau: np.multiply(calculate_PRMC(pred_rul, true_rul, tau, threshold, cost_reactive, cost_predictive, cost_rul), tau_distr(tau)), 0, np.inf, limit=limit)[0]

def _calculate_EPRMC_one_dim(pred_rul:np.ndarray, true_rul:np.ndarray, tau_distr, threshold:int, cost_reactive:float, cost_predictive:float, cost_rul:Union[float,np.ndarray], limit:int=150)->np.ndarray:
    """
    calculate EPRMC for multiple machines at one point in time

    Args:
        pred_rul (np.ndarray): pred
        true_rul (np.ndarray): trues
        tau_distr (functools.partial): leadtimes dist
        threshold (int): threshold
        cost_reactive (float): cost reactive
        cost_predictive (float): cost predictive
        cost_rul (Union[float,np.ndarray]): cost rul
        limit (int, optional): limit for integration. Defaults to 150.

    Returns:
        np.ndarray: EPRMC
    """
    out = np.zeros(pred_rul.shape)
    for machine in range(pred_rul.shape[0]):
        if true_rul[machine] == 0:
            out[machine:] = cost_reactive #if machine failed, set all following cells to the cost of failure
            break #break and go to next mach
        out[machine] = _calculate_EPRMC(pred_rul[machine], true_rul[machine], tau_distr, threshold, cost_reactive, cost_predictive, cost_rul, limit=limit)

    return out

def _calculate_EPRMC_two_dim(pred_rul:np.ndarray, true_rul:np.ndarray, tau_distr, threshold:int, cost_reactive:float, cost_predictive:float, cost_rul:Union[float,np.ndarray], limit:int=150)->np.ndarray:
    """
    calculate EPRMC for multiple machines at multiple points in time

    Args:
        pred_rul (np.ndarray): preds
        true_rul (np.ndarray): true
        tau_distr (functools.partial): lead times distribution
        threshold (int): threshold
        cost_reactive (float): cost reactive
        cost_predictive (float): cost predictive
        cost_rul (Union[float,np.ndarray]): cost rul
        limit (int, optional): limit for integration. Defaults to 150.

    Returns:
        np.ndarray: EPRMC
    """
    out = np.repeat(cost_reactive, pred_rul.shape[0]*pred_rul.shape[1]).reshape(pred_rul.shape)
    for machine in range(pred_rul.shape[0]):
        for iter, (p,t) in enumerate(zip(pred_rul[machine], true_rul[machine])):
            if t==0:
                break
            temp = _calculate_EPRMC(p,t,tau_distr=tau_distr,threshold=threshold,cost_reactive=cost_reactive,cost_predictive=cost_predictive,cost_rul=cost_rul[machine],limit=limit)
            out[machine,iter] = temp
    
    return out


def _expected_RUL(t:float,weib=partial(st.exponweib.pdf,a=2385.5255178273455,c=0.5602829765813255,loc=0,scale=5.013548199002173))->float:
    """
    Calculate expected RUL based on moment of maintenance and the initial distribution of useful lives
    Used for iPRMC

    Args:
        t (float): Moment of maintenance
        weib (functools.partial): distribution of useful lives. Defaults to partial(st.exponweib.pdf,a=2385.5255178273455,c=0.5602829765813255,loc=0,scale=5.013548199002173).

    Returns:
        float: expected rul
    """
    return integrate.quad(lambda x: weib(x+t)*x,0,np.inf)[0]/integrate.quad(lambda x: weib(x),t,np.inf)[0]

def calculate_iPRMC(pred:np.ndarray, true:np.ndarray, threshold:int, cost_reactive:float, cost_predictive:float, acquisition_cost:float, limit=150)->np.ndarray:
    """
    calculate iPRMC (PRMC during inference)

    Args:
        pred (np.ndarray): preds
        true (np.ndarray): trues
        threshold (int): threshold
        cost_reactive (float): cost reactive
        cost_predictive (float): cost predictive
        acquisition_cost (float): acquisition cost
        limit (int, optional): limit for integration. Defaults to 150.

    Returns:
        np.ndarray: PRMC during inference

    """
    stthresh = np.where(pred<=threshold) #note that this ignores machines that never have a prediction lower than the threshold (smaller than threshold)
    gtthresh = list(set(np.arange(0,pred.shape[0])).difference(set(stthresh[0]))) #these contain the machines that did not reach the threshold (greater than threshold, for all timesteps ==> Doomed to fail)
    moment_of_predicted_maintenance = np.array([(mach,np.min(stthresh[1][np.where(stthresh[0]==mach)])) for mach in set(stthresh[0])])
    zeros_true = np.where(true == 0)
    
    moment_of_failure = np.array([(mach,np.min(zeros_true[1][np.where(zeros_true[0]==mach)])) for mach in set(zeros_true[0])])
    moment_of_failure = np.delete(moment_of_failure,gtthresh,axis=0) #remove machines that are doomed to fail 
    expected_ruls = np.array([_expected_RUL(moment) for moment in moment_of_predicted_maintenance[:,1]])

    #calculate cost of predictive maintenance or reactive maintenance:
    out = np.where(moment_of_predicted_maintenance[:,1]>moment_of_failure[:,1],cost_reactive,cost_predictive+expected_ruls*(acquisition_cost/(expected_ruls+moment_of_predicted_maintenance[:,1])))
    #add rows that were doomed to fail:
    out = np.insert(out,gtthresh,[cost_reactive]*len(gtthresh))

    return out