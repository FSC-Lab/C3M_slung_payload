# System matrices for the slung payload system with variable length using Kane's method.
import torch
from .kinematics import get_Z, get_n, get_B, get_B_dot


class SlungPayloadVariableLengthSystem:
    def __init__(self, x_init):
        # Training parameters
        self.bs = 1
        self.num_dim_x = 12  # number of dimensions in state vector x
        self.num_dim_control = 4  # number of dimensions in control vector u
        
        # Slung payload parameters
        self.g = 9.81
        self.m_p = 0.5  # mass of the payload
        self.m_q = 1.63  # mass of the quadrotor
        
        # Kinematics
        self.Z = get_Z(x_init)
        self.n = get_n(x_init)
        self.n_T = self.n.transpose(1, 2)
        self.B = get_B(x_init)
        self.B_T = self.B.transpose(1, 2)
        self.B_dot = get_B_dot(x_init)
        
        # States
        self.r_tilde = torch.zeros(self.bs, 2, 1).type(x_init.type())  # (bs, 2, 1)
        self.r_q = torch.zeros(self.bs, 3, 1).type(x_init.type())
        self.l = torch.zeros(self.bs, 1, 1).type(x_init.type())  # (bs, 1, 1)
        self.v_tilde = torch.zeros(self.bs, 2, 1).type(x_init.type())  # (bs, 2, 1)
        self.v_q = torch.zeros(self.bs, 3, 1).type(x_init.type())  # (bs, 3, 1)
        self.l_dot = torch.zeros(self.bs, 1, 1).type(x_init.type())  # (bs, 1, 1)
        
        # MCG matrices
        self.M = torch.zeros(self.bs, 6, 6).type(x_init.type())
        self.C = torch.zeros(self.bs, 6, 6).type(x_init.type())
        self.F_g = torch.zeros(self.bs, 6, 1).type(x_init.type())
        
        # fx, gx
        self.fx = torch.zeros(self.bs, self.num_dim_x, 1).type(self.type)
        self.gx = torch.zeros(self.bs, self.num_dim_x, self.num_dim_control).type(self.type)
        

    def update_state(self, x):
        # Training parameters
        self.type = x.type()
        self.bs = x.shape[0]
        num_dim_x = self.num_dim_x
        
        # States
        r_tilde_x, r_tilde_y, r_q_x, r_q_y, r_q_z, l, v_tilde_x, v_tilde_y, v_q_x, v_q_y, v_q_z, l_dot = [x[:,i,0] for i in range(num_dim_x)]
        self.r_tilde = torch.stack([r_tilde_x, r_tilde_y], dim=1).unsqueeze(-1)
        self.r_q = torch.stack([r_q_x, r_q_y, r_q_z], dim=1).unsqueeze(-1)
        self.l = l.view(-1, 1, 1)
        self.v_tilde = torch.stack([v_tilde_x, v_tilde_y], dim=1).unsqueeze(-1)
        self.v_q = torch.stack([v_q_x, v_q_y, v_q_z], dim=1).unsqueeze(-1)
        self.l_dot = l_dot.view(-1, 1, 1)
        
        # Kinematics 
        self.Z = get_Z(self.r_tilde)
        self.n = get_n(self.r_tilde)
        self.n_T = self.n.transpose(1, 2)
        self.B = get_B(self.r_tilde)
        self.B_T = self.B.transpose(1, 2)
        self.B_dot = get_B_dot(self.r_tilde, self.v_tilde)
        
        # Update M, C, F_g
        self.calc_MCG()
        self.calc_fxgx()

    def get_M(self):
        return self.M
    
    def get_C(self):
        return self.C
    
    def get_G(self):
        return self.F_g
    
    def get_f(self):
        return self.fx

    def get_g(self):
        return self.gx

    # def get_DfDx(self):
    #     return .......

    def calc_MCG(self):

        # Compute M matrix
        M11 = self.m_p * (self.l ** 2) * torch.bmm(self.B_T, self.B)
        M12 = self.m_p * self.l * self.B_T
        M13 = torch.zeros(self.bs, 2, 1).type(self.type)
        M22 = (self.m_p + self.m_q) * torch.eye(3).repeat(self.bs, 1, 1).type(self.type)
        M23 = self.m_p * self.n
        M33 = self.m_p * torch.tensor([1]).repeat(self.bs, 1, 1).type(self.type)  # (bs, 1, 1)

        # Concactenate M matrix: (bs, 6, 6)
        self.M = torch.cat([
            torch.cat([M11, M12, M13], dim=2),
            torch.cat([M12.transpose(1, 2), M22, M23], dim=2),
            torch.cat([M13.transpose(1, 2), M23.transpose(1, 2), M33], dim=2)
        ], dim=1)

        # Compute C matrix
        C11 = (self.l ** 2) * torch.bmm(self.B_T, self.B_dot) + self.l_dot * self.l * torch.bmm(self.B_T, self.B)
        C12 = torch.zeros(self.bs, 2, 3).type(self.type)
        C13 = self.l * torch.bmm(torch.bmm(self.B_T, self.B), self.v_tilde)
        C21 = self.l * self.B_dot + self.B * self.l_dot
        C22 = torch.zeros(self.bs, 3, 3).type(self.type)
        C23 = torch.bmm(self.B, self.v_tilde)
        C31 = self.l * torch.bmm(self.n_T, self.B_dot)
        C32 = torch.zeros(self.bs, 1, 3).type(self.type)
        C33 = torch.zeros(self.bs, 1, 1).type(self.type)

        self.C = self.m_p * torch.cat([
            torch.cat([C11, C12, C13], dim=2),
            torch.cat([C21, C22, C23], dim=2),
            torch.cat([C31, C32, C33], dim=2)
        ], dim=1)

        # Compute G matrix
        G = torch.zeros(self.bs, 6, 3).type(self.type)
        G1 = self.m_p * self.l * self.B_T
        # G2 = (m_p + m_q) * torch.eye(3).repeat(bs, 1, 1).type(x.type())
        # G3 = m_p * n.transpose(1, 2)
        # Training for delta f and delta tau
        G2 = torch.zeros(self.bs, 3, 3).type(self.type)
        G3 = torch.zeros(self.bs, 1, 3).type(self.type)
        G = torch.cat([G1, G2, G3], dim=1)
        g_I = torch.tensor([0, 0, -self.g]).view(1, 3, 1).expand(self.bs, 3, 1).type(self.type)  # (bs, 3, 1)

        # Compute F_g
        self.F_g = torch.bmm(G, g_I)

    def calc_fxgx(self):
        # Compute fx
        vel = torch.cat([self.v_tilde, self.v_q, self.l_dot], dim=1)
        self.fx[:, 0:6, 0] = vel.squeeze(-1)
        self.fx[:, 6:12, 0] = torch.linalg.solve(self.M, self.F_g - torch.bmm(self.C, vel)).squeeze(-1)
        
        # Compute gx
        H = torch.zeros(self.bs, 6, self.num_dim_control).type(self.type)
        H[:, 2, 0] = 1  # Thrust force in x direction of earth frame
        H[:, 3, 1] = 1  # Thrust force in y direction of earth frame
        H[:, 4, 2] = 1  # Thrust force in z direction of earth frame
        H[:, 5, 3] = 1  # Extending torque
        
        gx_lower = torch.linalg.solve(self.M, H)
        self.gx = torch.cat([torch.zeros(self.bs, 6, self.num_dim_control).type(self.type), gx_lower], dim=1)
