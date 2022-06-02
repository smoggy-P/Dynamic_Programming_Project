class Policy(object):
    """Policy by dynamic programming

    Args:
        object (_type_): _description_
    """
    def __init__(self) -> None:
        pass
    
    def select_action(self, demond_state, power_supply_state):
        x, c = power_supply_state
        w = demond_state
        u = x - w
        return u