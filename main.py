from email import policy
from System import *
from Policy import *

num_simulation = 100

if __name__ == "__main__":
    plant = Plant()
    controller = Policy()
    
    state = plant.reset()
    for _ in range(num_simulation):
        u = controller.select_action(state)
        state = plant.step(u)
        print(state)