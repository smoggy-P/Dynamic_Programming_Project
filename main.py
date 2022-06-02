from email import policy
from System import *
from Policy import *

num_simulation = 100

if __name__ == "__main__":
    dt = 1 # one hour
    power_supply = Power_Supply(dt)
    demond_model = Demand(dt)
    controller = Policy()
    
    demond_state = demond_model.reset(0, init_month=10, init_day=1, init_hour=0)
    power_supply_state = power_supply.reset(init_battery=10)
    for _ in range(num_simulation):
        demond_state = demond_model.step()
        u = controller.select_action(demond_state, power_supply_state)
        power_supply_state = power_supply.step(u, real_w=0)  # peak, demond, battery
        print(power_supply_state, demond_state)