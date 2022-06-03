from demand_utils import *

class Battery(object):
    """Battery dynamics

    Attributes:
        cap (float): Maximum capacity of battery(kWh)
        alpha_c (float): Charging efficiency
        alpha_d (float): Discharging efficiency
        umax_c (float): Maximum charging rate(kW)
        umax_d (float): Maximum discharging rate(kW)
    """
    def __init__(self):
        self.c = None
        self.cap = 7.0
        self.alpha_c = 0.96
        self.alpha_d = 0.87
        self.umax_c = 4.0
        self.umax_d = 5.0
    
    def step(self, u, dt):
        """step dynamics for battery

        Args:
            u (float): charge rate
            dt (float): time

        Returns:
            float: amount of charge to be moved to/from(+/-) 
                   total amount of charge provided by power supplier(kW)
        """

        # clip the charge rate
        u = max(min(u, self.umax_c), -self.umax_d)

        # when charge to battery
        if u >= 0:
            c_hat = min(self.c + self.alpha_c * u * dt, self.cap)
            real_u = (c_hat - self.c)/self.alpha_c/dt 
        # when charge from battery
        else:
            c_hat = max(self.c + u * dt / self.alpha_d, 0)
            real_u = (c_hat - self.c)*self.alpha_d/dt
        self.c = c_hat
        return real_u
    
    def reset(self, init_c):
        self.c = init_c    


class Plant(object):
    """Dynamics for power supply
    Args:
        dt (float): time
        x (float): peak power
    """

    def __init__(self, dt):
        self.battery = Battery()
        self.dt = dt # one hour
        self.x = 0
        self.day = None
        self.hour = None
    
    def step(self, u, real_w):
        self.hour += self.dt
        if self.hour > 24:
            self.hour -= 24
            self.day += 1
        if self.day > 30:
            self.day = 0
            self.month += 1
        real_u = self.battery.step(u, self.dt)
        self.x = max(self.x, real_w + real_u)
        return [self.x, self.battery.c]

    def reset(self, init_battery, init_month, init_day, init_hour):
        self.month = init_month
        self.day = init_day
        self.hour = init_hour
        self.battery.reset(init_battery)
        self.x = 0
        return [self.x, self.battery.c]
        
        
