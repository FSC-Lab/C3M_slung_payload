# get Z, n, B, Bdot
import torch


def get_Z(r):
    Z = torch.sqrt(1-torch.bmm(r.transpose(1, 2), r))
    return Z


def get_n(r):
    Z = get_Z(r)
    n = torch.cat([r, -Z], dim=1)  # (bs, 3, 1)
    return n


def get_B(r):
    bs = r.shape[0]
    Z = get_Z(r)
    I2 = torch.eye(2).repeat(bs, 1, 1).type(r.type())  # shape (bs, 2, 2)
    B = torch.cat([I2, r.transpose(1, 2)/Z], dim=1)
    return B


def get_B_dot(r, v):
    bs = r.shape[0]
    O2 = torch.zeros(bs, 2, 2).type(r.type())  # shape (bs, 2, 2)
    Z = get_Z(r)
    r_T = r.transpose(1, 2)  # shape (bs, 1, 2)
    v_T = v.transpose(1, 2)  # shape (bs, 1, 2)
    B_dot = torch.cat([O2, (Z ** 2 * v_T + torch.bmm(r_T, v) * r_T)/(Z ** 3/2)], dim=1)  # shape (bs, 3, 2)
    return B_dot
 
