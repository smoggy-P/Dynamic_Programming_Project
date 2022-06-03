from email import policy
from System_copy import *
from Policy import *

num_simulation = 100

if __name__ == "__main__":
    dt = 1 # one hour
    plant = Plant(dt)
    controller = Policy()
    
    plant.reset(init_battery=10, init_month=10, init_day=1, init_hour=0)
    for _ in range(num_simulation):
        u = controller.select_action(plant)
        plant.step(u, real_w=1.2)  # peak, demond, battery
        print(plant.battery.c, u, plant.x)