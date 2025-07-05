import torch
import slung_payload_variable_length_utils as spu

num_dim_x = 12
num_dim_control = 4

bs = 1024  # Set up as constant for now, changed it to match main.py later
x_init = torch.tensor([[[0.], [0.], [0.], [0.], [0.], [2.], [0.], [0.], [0.], [0.], [0.], [0.]]]).repeat(bs, 1, 1)
slung_load_system = spu.SlungPayloadVariableLengthSystem(x_init)

def f_func(x):
    slung_load_system.update_state(x)
    f = slung_load_system.get_f()
    return f


def DfDx_func(x):
    raise NotImplemented('NotImplemented')


def B_func(x):
    slung_load_system.update_state(x)
    B = slung_load_system.get_g()
    return B


def DBDx_func(x):
    raise NotImplemented('NotImplemented')


def Bbot_func():
    Bbot = slung_load_system.get_Bbot()
    return Bbot
