from .rdn import make_rdn
from .swin import SwinIR
from .hat import HAT


def make(name):
    # if you use manual specification to make model,
    # please check each function of model class and parameters.

    model = None
    if name == 'RDN': model = make_rdn(6) 
    elif name == 'SwinIR': model = SwinIR()
    elif name == 'HAT': model = HAT()
    else:
        raise ValueErrors(f"[Error] There is no model {name}!")

    return model    
    