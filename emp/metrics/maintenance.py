from typing import Union
import tensorflow as tf
import numpy as np
import scipy.stats as st
import scipy.integrate as integrate
from functools import partial
from multiprocessing import Process, Queue
import warnings

from plotly.graph_objects import FigureWidget
from shiny import reactive

def _calculate_PRMC(pred: np.ndarray, true: np.ndarray, tau: int, threshold: int, cost_reactive: float, cost_predictive: float, cost_rul: Union[float, np.ndarray]) -> np.ndarray:
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
    pred_pos = np.repeat(np.inf, (pred.shape[0] * 2)).reshape((pred.shape[0], 2))
    for mach in machs:
        pred_pos[mach] = (
            (mach, np.amin(coords[coords.T[0] == mach].T[1]) + tau))  # add lead time

    # perceived failures per machine
    true_pos = np.array([(x, y) for x, y in enumerate(true.argmin(axis=1))])

    lost_rul = true_pos[:, 1] - pred_pos[:, 1]

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
            # if array, only look at the machines that
            for mach in prevented:
                cost[(mach[0], mach[1])] = cost_predictive + \
                    cost_rul[mach[0]] * lost_rul[mach[0]]
        else:
            for mach in prevented:
                cost[tuple(mach.astype(int))] = cost_predictive + \
                    cost_rul * lost_rul[int(mach[0])]

    return cost

def _calculate_PRMC_one_dim(pred_rul: np.ndarray, true_rul: np.ndarray, tau: int, threshold: int, cost_reactive: float, cost_predictive: float, cost_rul: Union[float, np.ndarray]) -> np.ndarray:
    """
    calculate prmc for one machine through time

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
    # schedule maintenance (PdM)
    maintenance = min(np.argwhere(pred_rul <= threshold)) + tau  # point in time at which maintenance is scheduled
    failure = min(np.argwhere(true_rul == 0))

    if failure >= maintenance:
        return cost_predictive + (failure - maintenance) * cost_rul
    else:
        return cost_reactive

def _calculate_PRMC_point(pred_rul: float, true_rul: float, tau: int, threshold: int, cost_reactive: float, cost_predictive: float, cost_rul: float) -> float:
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
        return cost_predictive + (true_rul - tau) * cost_rul

    return 0

def calculate_PRMC(pred: Union[float, np.ndarray], true: Union[float, np.ndarray], tau: int, threshold: int, cost_reactive: float, cost_predictive: float, cost_rul: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
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

def calculate_EPRMC(pred_rul: Union[float, np.ndarray], true_rul: Union[float, np.ndarray], threshold: int, cost_reactive: float, cost_predictive: float, cost_rul: Union[float, np.ndarray], limit: int=150, tau_distr=None, upper_bound=0.99, step_size=0, num_samples=0, tau_rvs=partial(st.lognorm.rvs, s=1.2, scale=6)) -> Union[float, np.ndarray]:
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
    if num_samples and step_size:
        warnings.warn("Only Monte Carlo integration will be done. Please select either num_samples or step_size, not both.", UserWarning)
    if (tau_distr is None):
        tau_distr = partial(st.lognorm.pdf, s=1.2, scale=6)
        upper_bound = st.lognorm.ppf(upper_bound, **tau_distr.keywords)
    else:
        if upper_bound < 1:
            warnings.warn(f'Are you certain that your upper bound is correct? Upperbound should correspond to a teststatistic. You submitted: {upper_bound}', UserWarning)

    if type(true_rul) == np.ndarray:
        if true_rul.ndim == 1:
            return _calculate_EPRMC_one_dim(pred_rul, true_rul, tau_distr, threshold, upper_bound, cost_reactive, cost_predictive, cost_rul, limit=limit, step_size=step_size, num_samples=num_samples, tau_rvs=tau_rvs)
        else:
            return _calculate_EPRMC_two_dim(pred_rul, true_rul, tau_distr, upper_bound, threshold, cost_reactive, cost_predictive, cost_rul, limit=limit, step_size=step_size, num_samples=num_samples, tau_rvs=tau_rvs)
    else:
        return _calculate_EPRMC(pred_rul, true_rul, tau_distr, upper_bound, threshold, cost_reactive, cost_predictive, cost_rul, limit=limit, step_size=step_size, num_samples=num_samples, tau_rvs=tau_rvs)

def _calculate_EPRMC(pred_rul: float, true_rul: float, tau_distr, upper_bound: float, threshold: int, cost_reactive: float, cost_predictive: float, cost_rul: float, limit: int=150, step_size=0, num_samples=0, tau_rvs=partial(st.lognorm.rvs, s=1.2, scale=6)) -> float:
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
# MC integration
    if num_samples:
        samples = np.zeros(num_samples)
        for i in range(num_samples):
            tau = tau_rvs()
            while tau > upper_bound:
                tau = tau_rvs()
            samples[i] = np.multiply(calculate_PRMC(pred_rul, true_rul, tau, threshold, cost_reactive, cost_predictive, cost_rul), tau_distr(tau))  # a lot of zeros
        return np.mean(samples)

# Riemann
    if step_size:
        x = np.arange(0, upper_bound, step_size)
        # y = np.array(list(map(tau_distr, x)))
        cost = np.array(list(map(lambda tau: np.sum(calculate_PRMC(pred_rul, true_rul, tau, threshold, cost_reactive, cost_predictive, cost_rul)), x)))
        return np.sum(cost * x)

# Integration
    return integrate.quad(lambda tau: np.multiply(calculate_PRMC(pred_rul, true_rul, tau, threshold, cost_reactive, cost_predictive, cost_rul), tau_distr(tau)), 0, upper_bound, limit=limit)[0]

def _calculate_EPRMC_one_dim(pred_rul: np.ndarray, true_rul: np.ndarray, tau_distr, upper_bound: float, threshold: int, cost_reactive: float, cost_predictive: float, cost_rul: Union[float, np.ndarray], limit: int=150, step_size=0, num_samples=0, tau_rvs=partial(st.lognorm.rvs, s=1.2, scale=6)) -> np.ndarray:
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
            out[machine:] = cost_reactive  # if machine failed, set all following cells to the cost of failure
            break  # break and go to next mach
        out[machine] = _calculate_EPRMC(pred_rul[machine], true_rul[machine], tau_distr, threshold, upper_bound, cost_reactive, cost_predictive, cost_rul, limit=limit, step_size=step_size, num_samples=num_samples, tau_rvs=tau_rvs)

    return out

def _calculate_EPRMC_two_dim(pred_rul: np.ndarray, true_rul: np.ndarray, tau_distr, upper_bound: float, threshold: int, cost_reactive: float, cost_predictive: float, cost_rul: Union[float, np.ndarray], limit: int=150, step_size=0, num_samples=0, tau_rvs=partial(st.lognorm.rvs, s=1.2, scale=6)) -> np.ndarray:
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

    Notes:
        This method indirectly makes a call to the underlying _calculate_PRMC_point method. It is advised to take a look at that method also.
    """

    out = np.repeat(0, pred_rul.shape[0] * pred_rul.shape[1]).reshape(pred_rul.shape)
    for machine in range(pred_rul.shape[0]):
        for iter, (p, t) in enumerate(zip(pred_rul[machine], true_rul[machine])):
            if t == 0:  # if the machine fails
                out[machine, iter] = cost_reactive  # due to machine failure, assign cost of failure
                break  # machine has failed, so no further calculations should be done for this machine, break the loop and go to the next machine
            if (t >= upper_bound) and (p > threshold):
                temp = 0
            else:
                temp = _calculate_EPRMC(p, t, tau_distr, upper_bound, threshold=threshold, cost_reactive=cost_reactive, cost_predictive=cost_predictive, cost_rul=cost_rul[machine], limit=limit, step_size=step_size, num_samples=num_samples, tau_rvs=tau_rvs)
            out[machine, iter] = temp
            if p <= threshold:  # if maintenance is scheduled, then stop calculation of costs for this machine
                break

    return out


def _expected_RUL(t: float, weib=partial(st.exponweib.pdf, a=2385.5255178273455, c=0.5602829765813255, loc=0, scale=5.013548199002173)) -> float:
    """
    Calculate expected RUL based on moment of maintenance and the initial distribution of useful lives
    Used for iPRMC

    Args:
        t (float): Moment of maintenance
        weib (functools.partial): distribution of useful lives. Defaults to partial(st.exponweib.pdf,a=2385.5255178273455,c=0.5602829765813255,loc=0,scale=5.013548199002173).

    Returns:
        float: expected rul
    """
    return integrate.quad(lambda x: weib(x + t) * x, 0, np.inf)[0] / integrate.quad(lambda x: weib(x), t, np.inf)[0]

def calculate_iPRMC(pred: np.ndarray, true: np.ndarray, threshold: int, cost_reactive: float, cost_predictive: float, acquisition_cost: float, limit=150) -> np.ndarray:
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
    stthresh = np.where(pred <= threshold)  # note that this ignores machines that never have a prediction lower than the threshold (smaller than threshold)
    gtthresh = list(set(np.arange(0, pred.shape[0])).difference(set(stthresh[0])))  # these contain the machines that did not reach the threshold (greater than threshold, for all timesteps ==> Doomed to fail)
    moment_of_predicted_maintenance = np.array([(mach, np.min(stthresh[1][np.where(stthresh[0] == mach)])) for mach in set(stthresh[0])])
    zeros_true = np.where(true == 0)

    moment_of_failure = np.array([(mach, np.min(zeros_true[1][np.where(zeros_true[0] == mach)])) for mach in set(zeros_true[0])])
    moment_of_failure = np.delete(moment_of_failure, gtthresh, axis=0)  # remove machines that are doomed to fail
    expected_ruls = np.array([_expected_RUL(moment) for moment in moment_of_predicted_maintenance[:, 1]])

    # calculate cost of predictive maintenance or reactive maintenance:
    out = np.where(moment_of_predicted_maintenance[:, 1] > moment_of_failure[:, 1], cost_reactive, cost_predictive + expected_ruls * (acquisition_cost / (expected_ruls + moment_of_predicted_maintenance[:, 1])))
    # add rows that were doomed to fail:
    out = np.insert(out, gtthresh, [cost_reactive] * len(gtthresh))

    return out


def multi_process_eprmc(thresholds, preds, trues, cost_reactive, cost_predictive, cost_rul, num_procs=10, upper_bound=0.95):
    num_thresholds = len(thresholds)
    queue_in = Queue(num_thresholds)
    queue_out = Queue(num_thresholds)
    procs = []

    list(map(queue_in.put, thresholds))  # add to queue

    for _ in range(num_procs):
        p = Process(target=_multi_process_eprmc, args=(queue_in, queue_out, preds, trues, cost_reactive, cost_predictive, cost_rul, upper_bound))
        procs.append(p)
        p.start()

    for p in procs:
        p.join()

    costs = np.zeros(num_thresholds)
    eprmc_dict = dict()
    while not queue_out.empty():
        eprmc_dict.update(queue_out.get())

    keys = list(eprmc_dict.keys())
    keys.sort()
    for index, key in enumerate(keys):
        costs[index] = eprmc_dict[key]

    return costs

def _multi_process_eprmc(queue_in, queue_out, preds, trues, cost_reactive, cost_predictive, cost_rul, upper_bound=0.95):
    while not queue_in.empty():
        thresh = queue_in.get()
        queue_out.put({thresh: np.sum(calculate_EPRMC(preds, trues, thresh, cost_reactive, cost_predictive, cost_rul, upper_bound=upper_bound))})
