from typing import Callable

import numpy as np
import control

class LinSysToIntegrate:
    nbr_states: int
    nbr_inputs: int
    nbr_outputs: int

    def _check_valid_state(self, x: np.ndarray):
        if x.shape != (self.nbr_states, 1):
            raise ValueError("Did not provide correct number of states.")

    def _check_valid_input(self, u: np.ndarray):
         if u.shape != (self.nbr_inputs, 1):
            raise ValueError("Did not provide correct number of inputs.")
            
    def _check_valid_output(self, y: np.ndarray):
        if y.shape != (self.nbr_outputs, 1):
            raise ValueError("Did not produce correct number of outputs.")

class StateSpaceToIntegrate(LinSysToIntegrate): # or, when shortened, ss2i
    ss : control.StateSpace

    def __init__(self, ss: control.StateSpace):
        self.ss = ss
        self.nbr_states = ss.A.shape[0]
        self.nbr_inputs = ss.B.shape[1]
        self.nbr_outputs = ss.C.shape[0]
    
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

class LinSysToIntegrateGenerator(LinSysToIntegrate):
    nbr_states: int
    nbr_inputs: int
    nbr_outputs: int
    def __init__(
        self, 
        list_state_space_to_integrate: list[StateSpaceToIntegrate], 
        nbr_inputs: int, 
        nbr_outputs: int
    ):
        self.nbr_states = sum([ss2i.nbr_states for ss2i in list_state_space_to_integrate])
        self.nbr_inputs = nbr_inputs
        self.nbr_outputs = nbr_outputs

    def compute_state_derivative_generator(
        self, 
        compute_state_derivative_x_u: Callable[[np.ndarray, np.ndarray], np.ndarray],
        compute_u_from_t: Callable[[float], np.ndarray],
    ):

        def compute_state_derivative(t: float, x: np.ndarray):
            u = compute_u_from_t(t)
            self._check_valid_input(u)
            self._check_valid_state(x)
            x_dot = compute_state_derivative_x_u(x, u)
            self._check_valid_state(x_dot)
            return x_dot

        return compute_state_derivative



if __name__ == '__main__':
    k = 2.  # kg
    c = .2  # Ns/m
    m = 0.1 # kg
    A_sys = np.array([[0.,1.],
                      [-k/m, -c/m]])
    B_sys = np.array([0.,
                     1/m])

    C_sys = np.array([[1.0, 0.0]])
    D_sys = np.array([[0.]])

    low_pass_cutoff_frequency = 500

    ss_sys = control.StateSpace(A_sys, B_sys, C_sys, D_sys)

    P_sys = control.ss2tf(ss_sys) # this is the transfer fucntion from input force to ouput position

    s = control.tf('s')
    
    G = 1 /(s + 500)

    G_P_inverse = control.minreal(G * P_sys^-1)

    ss_G = control.tf2ss(G)

    ss_G_P_inverse = control.tf2ss(G_P_inverse)

    ss2i_P = StateSpaceToIntegrate(P_sys)
    ss2i_G = StateSpaceToIntegrate(G)
    ss2i_G_P_inverse = StateSpaceToIntegrate(G_P_inverse)

    all_ss2i = [ss2i_P, ss2i_G, ss2i_G_P_inverse]

    fct_for_scipy_integrate_gen = LinSysToIntegrateGenerator(all_ss2i, 2, 3)


    def csd_xu_open_loop_sys_with_do(x: np.ndarray, u:np.ndarray):

        P_sys_x = x[0:ss2i_P.nbr_states, :]
        G_x = x[ss2i_P.nbr_states: ss2i_G.nbr_states, :]
        G_P_inverse_x = x[ss2i_G.nbr_states: ss_G_P_inverse.nbr_states, :]

        u_in = u[0, :]
        d = u[1, :]

        
        u_estim 











    