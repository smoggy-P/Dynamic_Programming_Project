class Policy(object):
    """Policy by dynamic programming

    Args:
        object (_type_): _description_
    """
    def __init__(self) -> None:
        pass
    
    def select_action(self, state):
        x, w, c= state
        u = x - w
        return u