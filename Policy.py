from demand_utils import *

class Estimation_from_time(object):
    """estimate energy demand for next time step

    Args:
        object (_type_): _description_

    Returns:
        _type_: _description_
    """
    def __init__(self):
        self.demand_model = Demand_Model(device = 'cpu')
        PATH = './demand_model_weights.pth'
        self.demand_model.load_state_dict(torch.load(PATH))
    
    def estimate(self, plant):
        # estimation of next time step
        hour = plant.hour + plant.dt
        day = plant.day
        month = plant.month
        if hour > 24:
            hour -= 24
            day = plant.day + 1
            if day > 30:
                day = 0
                month = plant.month + 1
        pro_month = data_preprocess(month, "Month")
        pro_day = data_preprocess(day, "Day")
        pro_hour = data_preprocess(hour, "Hour")
        features = torch.tensor([pro_month, pro_day, pro_hour], dtype=torch.float32)
        return self.demand_model.transition_step(features).item()
        

class Policy(object):
    """Policy by dynamic programming

    Args:
        object (_type_): _description_
    """
    def __init__(self):
        self.estimator = Estimation_from_time()
    
    def select_action(self, plant):
        w_hat = self.estimator.estimate(plant)
        u = plant.x - w_hat
        return u, w_hat