from zmq import device
import joblib
import torch
from torch.autograd import Variable
import numpy as np
from LSTM import LSTM

class Estimator(object):
    """estimate energy demand for next time step

    Args:
        object (_type_): _description_

    Returns:
        _type_: _description_
    """
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.demand_model = torch.load("demand_esti_model.pkl", map_location=self.device)
        self.demand_model.eval()
        self.data_scaler = joblib.load("data_scaler.pkl")
    
    def estimate(self, x_pre):
        # estimation of next time step
        a = self.data_scaler.transform(np.array([[x_pre]])).reshape(-1,1,1)
        return self.data_scaler.inverse_transform(self.demand_model(Variable(torch.Tensor(a)).cuda()).cpu().data.numpy())[0,0]

class Policy(object):
    """Policy by dynamic programming

    Args:
        object (_type_): _description_
    """
    def __init__(self):
        self.estimator = Estimator()
    
    def select_action(self, plant):
        l_hat = self.estimator.estimate(plant.l)
        u = plant.p - l_hat
        return u, l_hat