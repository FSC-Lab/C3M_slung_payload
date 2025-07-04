import torch
import slung_payload_variable_length_utils as spu

num_dim_x = 12
num_dim_control = 4

g = 9.81
m_p = 0.5  # mass of the payload
m_q = 1.63  # mass of the quadrotor

# Add another function to get dynamics simplify f_func, B_func

slung_load_system = spu.SlungPayloadVariableLengthSystem(dd, sss)


def f_func(x):
    slung_load_system.update_state()
    f = slung_load_system.get_f()
    return f


def DfDx_func(x):
    raise NotImplemented('NotImplemented')


def B_func(x):
    
    
    return B


def DBDx_func(x):
    raise NotImplemented('NotImplemented')


# How to calculate Bbot: null space of B
def Bbot_func(x):
    # Bbot: bs x n x (n-m)
    bs = x.shape[0]
    Bbot = torch.zeros(bs, num_dim_x, num_dim_x-num_dim_control).type(x.type())
    Bbot[:, 0, 0] = 1
    Bbot[:, 1, 1] = 1
    Bbot[:, 2, 2] = 1
    Bbot[:, 3, 3] = 1
    Bbot[:, 4, 4] = 1
    Bbot[:, 5, 5] = 1
    
    return Bbot
