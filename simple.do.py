from typing import Callable

import numpy as np
import control

class StateSpaceToIntegrate: # or, when shortened, ss2i
    nbr_states: int
    nbr_inputs: int
    nbr_outputs: int
    ss : control.StateSpace

    def __init__(self, ss: control.StateSpace):
        self.ss = ss
        self.nbr_states = ss.A.shape[0]
        self.nbr_inputs = ss.B.shape[1]
        self.nbr_outputs = ss.C.shape[0]

    def _check_valid_state(self, x: np.ndarray):
        if x.shape != (self.nbr_states, 1):
            raise ValueError("Did not provide correct number of states.")

    def _check_valid_input(self, u: np.ndarray):
         if u.shape != (self.nbr_inputs, 1):
            raise ValueError("Did not provide correct number of inputs.")
            
    def _check_valid_output(self, y: np.ndarray):
        if y.shape != (self.nbr_outputs, 1):
            raise ValueError("Did not produce correct number of outputs.")
    
    def compute_state_derivative(self, x: np.ndarray, u: np.ndarray):
        self._check_valid_state(x)
        self._check_valid_input(u)

        x_dot = self.ss.A @ x + self.ss.B @ u
        self._check_valid_state(x_dot)
        return x_dot

    def compute_output(self, x : np.ndarray, u: np.ndarray):
        self._check_valid_state(x)
        self._check_valid_input(u)

        y = self.ss.C @ x + self.ss.D @ u
        self._check_valid_output(y)
        return y

    def generate_fction_to_integrate(
        self,
        compute_u_from_t: Callable[[float], np.ndarray],
    ):

        def fction_to_integrate(t: float, x: np.ndarray):
            u = compute_u_from_t(t)
            x_dot = self.compute_state_derivative(x, u)
            return x_dot

        return fction_to_integrate

    def recreate_output(
        self, 
        compute_u_from_t: Callable[[float], np.ndarray],
        t_arr: np.ndarray, 
        x_arr: np.ndarray,
    ):
    # Have to use np.vectorize with signature argument somewhat
        pass
    



if __name__ == '__main__':
    k = 2.  # kg
    c = .2  # Ns/m
    m = 0.1 # kg
    A_n = np.array([[0.,1.],
                      [-k/m, -c/m]])
    B_n = np.array([0.,
                     1/m])

    C_n = np.array([[1.0, 0.0]])
    D_n = np.array([[0.]])

    low_pass_cutoff_frequency = 500
    low_pass_damping_coeff = 0.707

    ss_n = control.StateSpace(A_n, B_n, C_n, D_n)

    P_n = control.ss2tf(ss_n) # this is the transfer fucntion from input force to ouput position

    s = control.tf('s')
    
    Q = low_pass_cutoff_frequency**2 /(s**2 + 2*low_pass_damping_coeff*low_pass_cutoff_frequency*s + low_pass_cutoff_frequency**2)

    Q_P_inverse = control.minreal(Q / P_n)

    ss_Q = control.tf2ss(Q)

    ss_Q_P_inverse = control.tf2ss(Q_P_inverse)

    A_q = ss_Q.A
    B_q = ss_Q.B
    C_q = ss_Q.C
    D_q = ss_Q.D

    A_c = ss_Q_P_inverse.A
    B_c = ss_Q_P_inverse.B
    C_c = ss_Q_P_inverse.C
    D_c = ss_Q_P_inverse.D

    M_inv = np.linalg.inv(np.eye(D_q.shape(0)) - D_q + D_c @ D_n)

    M_inv_D_c_C_n = (M_inv @ D_c @ C_n)
    M_inv_C_q = (M_inv @ C_q)
    M_inc_C_c = (M_inv @ C_c)

    C_n_m_D_n_M_inv_D_c_C_n = (C_n - D_n @ M_inv_D_c_C_n)
    D_n_M_inv_C_q = (D_n @ M_inv_C_q)
    D_n_M_inc_C_c = (D_n @ M_inc_C_c)

    A_tot = np.block([
        [A_n - B_n @ M_inv_D_c_C_n,     B_n @ M_inv_C_q,       -B_n @ M_inc_C_c],
        [-B_q @ M_inv_D_c_C_n,          A_q + B_q @ M_inv_C_q, -B_q @ M_inc_C_c],
        [B_c @ C_n_m_D_n_M_inv_D_c_C_n, B_c @ D_n_M_inv_C_q ,  -B_c @ D_n_M_inc_C_c],
    ])

    M_inv_D_c_Dn = (M_inv @ D_c @ D_n)
    I_m_M_inv_D_c_Dn = np.eye(M_inv_D_c_Dn.shape(0)) - M_inv_D_c_Dn
    M_inv_D_q = M_inv @ D_q
    I_m_M_inv__D_c_Dn_m_Dq_ = I_m_M_inv_D_c_Dn + M_inv_D_q
    D_n_I_m_M_inv__D_c_Dn_m_Dq_ = D_n @ I_m_M_inv__D_c_Dn_m_Dq_
    D_n_I_m_M_inv_D_c_Dn = D_n @ I_m_M_inv_D_c_Dn

    B_tot = np.block([
        [B_n @ I_m_M_inv__D_c_Dn_m_Dq_ ,     B_n @ I_m_M_inv_D_c_Dn    ],
        [B_q @ I_m_M_inv__D_c_Dn_m_Dq_ ,     -B_q @ M_inv_D_c_Dn       ],
        [B_c @  D_n_I_m_M_inv__D_c_Dn_m_Dq_, B_c @ D_n_I_m_M_inv_D_c_Dn],
    ])

    C_tot = np.block([
        [C_n_m_D_n_M_inv_D_c_C_n, D_n_M_inv_C_q, -D_n_M_inc_C_c],
        [M_inv_D_c_C_n,           -M_inv_C_q,    M_inc_C_c     ],
    ])

    D_tot = np.block([
        [D_n_I_m_M_inv__D_c_Dn_m_Dq_, D_n_I_m_M_inv_D_c_Dn],
        [M_inv_D_c_Dn - M_inv_D_q,    M_inv_D_c_Dn],
    ])


    ss_tot = control.StateSpace(A_tot, B_tot, C_tot, D_tot)

    ss2i_tot = StateSpaceToIntegrate(ss_tot)










    