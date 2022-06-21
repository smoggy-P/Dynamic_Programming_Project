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
    
    def step(self, u, dt):
        """step dynamics for battery

        Args:
            u (float): charge rate
            dt (float): time

        Returns:
            float: amount of charge to be moved to/from(+/-) 
                   total amount of charge provided by power supplier(kW)
        """
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
        p (float): peak power
        l (float): load
    """

    def __init__(self, dt):
        self.battery = Battery()
        self.p = 0
        self.l = 0
        self.dt = dt
    
    def step(self, u, real_l):
        real_u = self.battery.step(u, self.dt)
        self.p = max(self.p, real_l + real_u)   #TODO check real_u
        self.l = real_l 
        return [self.p, self.battery.c, self.l]

    def reset(self, init_battery, init_peak, init_load):
        self.battery.reset(init_battery)
        self.p = init_peak
        self.l = init_load
        return [self.p, self.battery.c, self.l]
