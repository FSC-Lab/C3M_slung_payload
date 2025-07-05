# get Z, n, B, Bdot
import torch


def get_Z(r_p):
    Z = -torch.sqrt(1-torch.bmm(r_p.transpose(1, 2), r_p))
    return Z


def get_n(r_p):
    Z = get_Z(r_p)
    n = torch.cat([r_p, Z], dim=1)  # (bs, 3, 1)
    return n


def get_B(r_p):
    bs = r_p.shape[0]
    Z = get_Z(r_p)
    I2 = torch.eye(2).repeat(bs, 1, 1).type(r_p.type())  # shape (bs, 2, 2)
    B = torch.cat([I2, -r_p.transpose(1, 2)/Z], dim=1)
    return B


def get_B_dot(r_p, v_p):
    bs = r_p.shape[0]
    O2 = torch.zeros(bs, 2, 2).type(r_p.type())  # shape (bs, 2, 2)
    Z = get_Z(r_p)
    r_p_T = r_p.transpose(1, 2)  # shape (bs, 1, 2)
    v_p_T = v_p.transpose(1, 2)  # shape (bs, 1, 2)
    B_dot = torch.cat([O2, -(Z ** 2 * v_p_T + torch.bmm(r_p_T, v) * r_p_T)/(Z ** 3)], dim=1)  # shape (bs, 3, 2)
    return B_dot
