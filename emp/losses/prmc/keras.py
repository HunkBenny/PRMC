
import tensorflow as tf
from keras.losses import Loss
import numpy as np

class PRMCLoss(Loss):
    """
    class for keras PRMCLOSS
    """

    def __init__(self, cost_reactive:float, cost_predictive:float, cost_rul:float=None, name:str=None):
        """
        instantiate Loss function for keras models

        Args:
            cost_reactive (float): CR
            cost_predictive (float): CP
            shared_cost_rul (float, optional): Should be a number, used for shared costs. Do not supply if planning on using individual costs. Defaults to None. Not an array
            name (str, optional): Name. Defaults to None.
        """
        self.cost_reactive = cost_reactive
        self.cost_predictive = cost_predictive
        self.shared_cost_rul = cost_rul

        super().__init__(name=name)

    def __call__(self, y_true:np.ndarray, y_pred:np.ndarray, ind_cost_rul:np.ndarray=None)-> np.ndarray:
        """
        call object to calculate loss

        Args:
            y_true (np.ndarray): true
            y_pred (np.ndarray): pred
            ind_cost_rul (np.ndarray, optional): Only supply this if you want to use individual costs of RUL during training. Defaults to None.

        Returns:
            np.ndarray: loss
        """
        diff = y_true-y_pred
        sq = diff**2
        if ind_cost_rul == None and self.shared_cost_rul == None:
            raise ValueError(
                f"Either ind_cost_rul or cost_rul needs to be not None-type.")

        if ind_cost_rul != None:
            return tf.where(y_pred < 0, sq*(ind_cost_rul)+self.cost_predictive, tf.where(diff > 0,  sq*(ind_cost_rul)+self.cost_predictive, self.cost_reactive*sq+self.cost_predictive))
        else:
            return tf.where(y_pred < 0, sq*(self.shared_cost_rul)+self.cost_predictive, tf.where(diff > 0,  sq*(self.shared_cost_rul)+self.cost_predictive, self.cost_reactive*sq+self.cost_predictive))


class MSEOverestimate(Loss):
    """
    Class for custom MSE loss
    """

    def __init__(self, cost_overestimate:int=1):
        """
        instantiate custom loss

        Args:
            cost_overestimate (int, optional): only supply if you intend on punishing the overestimation of the true label. Defaults to 1 (no punishment).
        """
        self.cost_overestimate = cost_overestimate
        super().__init__()

    def __call__(self, y:np.ndarray, pred:np.ndarray,**kwargs)->tf.TensorArray:
        """
        Call the object to calculate loss

        Args:
            y (np.ndarray): true
            pred (np.ndarray): pred

        Returns:
            tf.TensorArray: tensor containing loss values
        """
        diff = pred - y
        return tf.where(diff > 0, self.cost_overestimate*diff**2, diff**2) #if diff>0 ==> pred > true