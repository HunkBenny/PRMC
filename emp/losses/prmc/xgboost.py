import numpy as np
import xgboost as xgb
from typing import Union, Tuple

class XGBObjectiveFunction:
  """
  class for xgb prmcloss
  """
  def __init__(self, cost_reactive:float, cost_predictive:float, cost_rul_train: Union[float,np.ndarray], cost_rul_eval: Union[float,np.ndarray] =None, name=None):
        """instantiate PRMCLOSS Objective-function object

        Args:
            cost_reactive (float): cost of reactive maintenance
            cost_predictive (float): cost of predictive maintenance
            cost_rul_train (Union[float,np.ndarray]): cost RUL train. Either a float or an array containing a value for every single datapoint. Defaults to None.
            cost_rul_eval (Union[float,np.ndarray], optional): cost RUL eval. Either a float or an array containing a value for every single datapoint. Defaults to None.
            name (_type_, optional): name for the model. Defaults to None.

        Raises:
            ValueError: _description_
        """
        self.cost_reactive = cost_reactive
        self.cost_predictive = cost_predictive
        self.cost_rul_train = cost_rul_train
        if type(cost_rul_train) == np.ndarray:
            print("Initialized objective function with individual costs")
        else:
            print("Initialized objective function with shared costs")

        ### what to do if no eval cost:
        if type(cost_rul_eval) == np.ndarray:
            self.cost_rul_eval = cost_rul_eval
        elif cost_rul_eval == None:
            self.cost_rul_eval  = None
        else:
            if type(cost_rul_train) == np.ndarray:
                raise ValueError("Please provide a value for cost_rul_eval. If cost_rul_train is an array, then preferrably supply an array or cost_rul_eval")
            else:
                self.cost_rul_eval = cost_rul_train

  def _gradient(self, pred:np.ndarray, y:np.ndarray)->np.ndarray:
    """calculates gradient PRMCLOSSS

    Args:
        pred (np.ndarray): prediction
        y (np.ndarray): true

    Returns:
        np.ndarray: gradient
    """
    return np.where(pred <= y, 4*(pred-y)**3*self.cost_rul_train, 4*(pred-y)**3*self.cost_reactive)

  def _hessian(self, pred:np.ndarray, y:np.ndarray)->np.ndarray:
    """calculates hessian PRMCLOSSS

    Args:
        pred (np.ndarray): prediction
        y (np.ndarray): true

    Returns:
        np.ndarray: hessian
    """
    return np.where(pred <= y, 12*(pred-y)**2*self.cost_rul_train, 12*(pred-y)**2*self.cost_reactive)

  def __call__(self, pred:np.ndarray, dtrain:np.ndarray)->Tuple[np.ndarray,np.ndarray]:
    """
    calculate grad & hess when object is called

    Args:
        pred (_type_): pred
        dtrain (_type_): true

    Returns:
        Tuple[np.ndarray,np.ndarray]: return grad & hess
    """
    y= dtrain.get_label()
    grad = self._gradient(pred,y)
    hess = self._hessian(pred,y)
    return grad, hess

  def metric(self, pred:np.ndarray, dtrain:xgb.DMatrix)->Tuple[str,np.ndarray]:
      """
      Calculate metric to be returned at the end of a train step

      Args:
          pred (np.ndarray): prediction
          dtrain (xgb.DMatrix): dmatrix with data

      Returns:
          tuple: metric evaluation
      """
      y=dtrain.get_label()
      #distinguish between train and val rul
      if self.cost_rul_eval == None:
          return 'PRMCL', np.mean(np.where(pred<=y,(y-pred)*np.mean(self.cost_rul_train)+self.cost_predictive,self.cost_reactive))

      #if eval cost is supplied: try to find whether pred is for train or for eval dataset
      if pred.shape[0] == self.cost_rul_train.shape[0]:
          return 'PRMCL', np.mean(np.where(pred<=y,(y-pred)*np.mean(self.cost_rul_train)+self.cost_predictive,self.cost_reactive))
      else:
          return 'PRMCL', np.mean(np.where(pred<=y,(y-pred)*np.mean(self.cost_rul_eval)+self.cost_predictive,self.cost_reactive))


class XGBMSE:
    """
    Class for xgboost mse loss
    """
    def __init__(self,cost_overestimation=1):
        """
        Instantiate MSE loss objective function
        """
        self.cost_overestimation = cost_overestimation

    def _gradient(self, pred:np.ndarray, y:np.ndarray)->np.ndarray:
        """
        calculate gradient

        Args:
            pred (np.ndarray): pred
            y (np.ndarray): true

        Returns:
            np.ndarray: grad
        """
        diff = (pred-y)
        return np.where(diff > 0, self.cost_overestimation * 2 * diff , 2 * diff)

    def _hessian(self, pred:np.ndarray, y:np.ndarray)->np.ndarray:
        """calculate hessian

        Args:
            pred (np.ndarray): pred
            y (np.ndarray): true

        Returns:
            np.ndarray: hess
        """
        diff = (pred-y)
        return np.where(diff > 0, self.cost_overestimation * 2, 2) #hess is equal to 2 for mse

    def __call__(self, pred:np.ndarray, dtrain:xgb.DMatrix)->Tuple[np.ndarray,np.ndarray]:
        """
        call loss object

        Args:
            pred (np.ndarray): pred
            dtrain (xgb.DMatrix): true

        Returns:
            Tuple[np.ndarray,np.ndarray]: returns grad and hess
        """
        y=dtrain.get_label()
        grad = self._gradient(pred,y)
        hess = self._hessian(pred,y)
        return grad, hess

    def metric(self, pred:np.ndarray, dtrain:xgb.DMatrix)->Tuple[str,np.ndarray]:
        """
        metric after each trainstep

        Args:
            pred (np.ndarray): pred
            dtrain (xgb.DMatrix): true

        Returns:
            Tuple[str,np.ndarray]: _description_
        """
        y=dtrain.get_label()
        diff = (pred-y)
        return 'MSE', np.mean(np.where(diff > 0, self.cost_overestimation * diff**2 , diff**2))





class XGBPseudoHuberLoss:
    """
    XGBPSEUDOHUBERLOSS class
    Adapted from:
    https://github.com/dmlc/xgboost/issues/5479
    """
    def __init__(self):
        pass

    def _gradient(self, pred, y):
        pass

    def _hessian(self, pred, y):
        pass

    def __call__(self, pred:np.ndarray, dtrain:xgb.DMatrix)->Tuple[np.ndarray,np.ndarray]:
        """        
        call the pseudohuber object

        Note that all calculations are done within this method

        Args:
            pred (np.ndarray): pred
            dtrain (xgb.DMatrix): true

        Returns:
            Tuple[np.ndarray,np.ndarray]: grad and hess
        """
        y_true=dtrain.get_label()
        z = pred - y_true
        delta = 1
        scale = 1 + (z/delta)**2
        scale_sqrt = np.sqrt(scale)
        grad = z/scale_sqrt
        hess = 1/(scale*scale_sqrt)
        return grad, hess
    
    def metric(self, pred:np.ndarray, dtrain:xgb.DMatrix)->Tuple[str,np.ndarray]:
        """
        metric at the end of trainstep

        Args:
            pred (np.ndarray): pred
            dtrain (xgb.DMatrix): true

        Returns:
            Tuple[str,np.ndarray]: output metric
        """
        y = dtrain.get_label()
        d = (pred-y)

        delta = 1
        scale = 1 + (d/delta) ** 2

        return 'Huber', np.mean(np.where(abs(d)<=delta,(d**2)/2,delta * (np.abs(d)-delta/2)))