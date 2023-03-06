from typing import Callable
from dataclasses import dataclass

import numpy as np
import control
import scipy.integrate as integrate
import matplotlib.pyplot as plt

@dataclass
class ProcessModel:
    nbr_states: int
    nbr_inputs: int
    nbr_outputs: int
    fct_for_x_dot: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    fction_for_y: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]

class ProcessModelToIntegrate(ProcessModel): # or, when shortened, pm2i
    def _check_valid_state(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] != self.nbr_states:
            raise ValueError(f"Did not provide correct number of states. System has {self.nbr_states} states, {x.shape} states were provided.")
        return np.reshape(x, (self.nbr_states, 1))

    def _check_valid_input(self, u: np.ndarray) :
        if u.shape != (self.nbr_inputs, 1):
            raise ValueError(f"Did not provide correct number of inputs. System has {self.nbr_inputs} inputs, {u.shape} inputs were provided.")
            
    def _check_valid_output(self, y: np.ndarray):
        if y.shape != (self.nbr_outputs, 1):
            raise ValueError(f"Did not produce correct number of outputs. System has {self.nbr_outputs} outputs, {y.shape} outputs were provided.")
    
    def compute_state_derivative(self, x: np.ndarray, u: np.ndarray):
        x = self._check_valid_state(x)
        self._check_valid_input(u)
        # with warnings.catch_warnings():
        #     warnings.filterwarnings('error')
        #     try:
        x_dot = self.ss.A @ x + self.ss.B @ u
            # except RuntimeWarning:
            #     print(f'{self.ss.A=}')
            #     print(f'{x=}') 
            #     print(f'{self.ss.B=}') 
            #     print(f'{u=}') 
            #     raise ValueError
        x_dot = self._check_valid_state(x_dot)
        return x_dot

    def compute_output(self, x : np.ndarray, u: np.ndarray):
        x = self._check_valid_state(x)
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
            return x_dot.ravel()

        return fction_to_integrate

    def recreate_input_and_output(
        self, 
        compute_u_from_t: Callable[[float], np.ndarray],
        t_arr: np.ndarray, 
        x_arr: np.ndarray,
    ):
        u_arr = np.zeros(shape=(self.nbr_inputs, t_arr.shape[0]))
        y_arr = np.zeros(shape=(self.nbr_outputs, t_arr.shape[0]))

        for i, t in enumerate(t_arr):
            u = compute_u_from_t(t)
            u_arr[:, i] = u.ravel()
            y_arr[:, i] = self.compute_output(x_arr[:,i], u).ravel()

        return u_arr, y_arr

class DisturbanceObservedProcessModelGenerator: # or, when shortened, pm2i
    # Only works when original process model only returns full state
    nbr_states: int
    nbr_inputs: int
    nbr_disturbances : int
    f: Callable[[np.ndarray, ], np.ndarray]
    g_1: Callable[[np.ndarray, ], np.ndarray]
    g_2: Callable[[np.ndarray, ], np.ndarray]
    h: Callable[[np.ndarray, ], np.ndarray]
    L_d: Callable[[np.ndarray, ], np.ndarray]
    p: Callable[[np.ndarray, ], np.ndarray]

    nbr_total_states: int
    nbr_total_inputs: int
    nbr_total_outputs: int

    def __init__(
        self, 
        nbr_states: int,
        nbr_inputs: int,
        nbr_disturbances : int,
        f: Callable[[np.ndarray, ], np.ndarray],
        g_1: Callable[[np.ndarray, ], np.ndarray],
        g_2: Callable[[np.ndarray, ], np.ndarray],
        L_d: Callable[[np.ndarray, ], np.ndarray],
        p: Callable[[np.ndarray, ], np.ndarray],
    ):
        self.nbr_states = nbr_states
        self.nbr_inputs = nbr_inputs
        self.nbr_disturbances = nbr_disturbances
        self.f = f
        self.g_1 = g_1
        self.g_2 = g_2
        self.L_d = L_d
        self.p = p

        self.nbr_total_states = nbr_states + nbr_disturbances
        self.nbr_total_inputs = nbr_inputs + nbr_disturbances
        self.nbr_total_outputs = nbr_states + nbr_disturbances

    def _check_valid_state(self, x: np.ndarray):
        if x.shape != (self.nbr_states, 1):
            raise ValueError(f"Did not provide correct number of states. System has {self.nbr_states} states, {x.shape} states were provided.")

    def _check_valid_input(self, u: np.ndarray) :
        if u.shape != (self.nbr_inputs, 1):
            raise ValueError(f"Did not provide correct number of inputs. System has {self.nbr_inputs} inputs, {u.shape} inputs were provided.")

    def _check_valid_disturbance(self, d: np.ndarray):
        if d.shape != (self.nbr_disturbances, 1):
            raise ValueError(f"Did not provide correct number of disturbances. System has {self.nbr_disturbances} states, {d.shape} states were provided.")
    
    def _check_valid_total_state(self, total_state: np.ndarray) -> np.ndarray:
        if total_state.shape[0] != self.nbr_total_states:
            raise ValueError(f"Did not provide correct number of states. System has {self.nbr_total_states} states, {total_state.shape} states were provided.")
        return np.reshape(total_state, (self.nbr_states, 1))

    def _check_valid_total_input(self, total_input: np.ndarray) :
        if total_input.shape != (self.nbr_total_inputs, 1):
            raise ValueError(f"Did not provide correct number of inputs. System has {self.nbr_total_inputs} inputs, {total_input.shape} inputs were provided.")

    def _check_valid_total_output(self, total_output: np.ndarray) :
        if total_output.shape != (self.nbr_total_outputs, 1):
            raise ValueError(f"Did not provide correct number of outputs. System has {self.nbr_total_outputs} outputs, {total_output.shape} outputs were provided.")

    def compute_x_dot_process_model(
        self,
        x: np.ndarray, 
        u: np.ndarray, 
        d:np.ndarray, 
        d_estim: np.ndarray
    ) -> np.ndarray:

        self._check_valid_state(x)
        self._check_valid_input(u)
        self._check_valid_disturbance(d)
        self._check_valid_disturbance(d_estim)

        x_dot = self.f(x) + self.g_1(x) @ u + self.g_2(x) @ d - self.g_2(x) @ d_estim

        self._check_valid_state(x_dot)

        return x_dot

    def compute_z_dot_disturb_observer(self, x: np.ndarray, z:np.ndarray, u: np.ndarray):
        self._check_valid_state(x)
        self._check_valid_disturbance(z)
        self._check_valid_input(u)

        z_dot = -self.L_d(x) @ self.g_2(x) @ z - self.L_d(x) @ ( self.f(x) + self.g_1(x) @ u + self.g_2(x) @ self.p(x))

        self._check_valid_disturbance(z_dot)

        return z_dot

    def compute_state_derivative(self, total_state: np.ndarray, total_input: np.ndarray) -> np.ndarray:
        total_state = self._check_valid_total_state(total_state)
        self._check_valid_total_input(total_input)
        
        x = total_state[0:self.nbr_states, 0]
        z = total_state[self.nbr_states:self.nbr_states+self.nbr_disturbances, 0]

        u = total_input[0:self.nbr_inputs, 0]
        d = total_state[self.nbr_inputs:self.nbr_inputs+self.nbr_disturbances, 0]

        z_dot = self.compute_z_dot_disturb_observer(x, z, u)

        d_estim = z + self.p(x)

        x_dot = self.compute_x_dot_process_model(x, u, d, d_estim)

        total_state_dot = np.block([
            [x_dot],
            [z_dot]
        ])

        total_state_dot = self._check_valid_total_state(total_state_dot)

        return total_state_dot

    def compute_output(self, total_state: np.ndarray) -> np.ndarray:
        total_state = self._check_valid_total_state(total_state)
        
        x = total_state[0:self.nbr_states, 0]
        z = total_state[self.nbr_states:self.nbr_states+self.nbr_disturbances, 0]

        d_estim = z + self.p(x)

        total_output = np.block([
            [x],
            [d_estim]
        ])

        self._check_valid_total_output(total_output)

        return total_output

    def generate_process_model(self):
        return ProcessModel(self.nbr_total_states, self.nbr_total_inputs, self.nbr_total_outputs, self.compute_state_derivative, self.compute_output)
         

if __name__ == '__main__':
    k_1 = 0.9
    k_2 = 0.02
    c = 0.2  # Ns/m
    m = 1 # kg

    # observer parameters

    alpha_d = 5
    K1_d = 10


    def f(x:np.ndarray) -> np.ndarray:
        x1 = x[0,0]
        x2 = x[1,0]

        return np.ndarray([
            [x2],
            [f.csnt_0 * x2 + f.cnst_1 * x1 + f.csnt_2 * x1*x1 ]
        ])

    f.csnt_0 = -c/m
    f.csnt_1 = -k_1/m
    f.csnt_2 = -k_2/m

    def g_1(x:np.ndarray) -> np.ndarray:
        return g_1.cnst_0

    g_1.cnst_0  = np.ndarray([
                    [0],
                    [1/m]
                ])

    def g_2(x:np.ndarray) -> np.ndarray:
        return g_1(x)

    def L_d(x:np.ndarray) -> np.ndarray:
        return L_d.cnst_0

    L_d.csnt_0 = np.ndarray([
        [0, m*alpha_d]
    ])

    def p(x:np.ndarray) -> np.ndarray:
        x2 = x[1,0]
        return p.cnst_0 * x2 + p.cnst_1

    p.cnst_0 = m*alpha_d
    p.cnst_1 = K1_d

    dopmg = DisturbanceObservedProcessModelGenerator(2, 1, 1, f, g_1, g_2, L_d, p)

    pm = dopmg.generate_process_model()


    



