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
    fct_for_x_dot: Callable[[np.ndarray, np.ndarray], np.ndarray]
    fction_for_y: Callable[[np.ndarray, np.ndarray], np.ndarray]

@dataclass
class ProcessModelToIntegrate(ProcessModel): # or, when shortened, pm2i
    ran_checks_for_compute_state_derivative: bool
    ran_checks_for_compute_output: bool

    def __init__(
        self,
        nbr_states: int,
        nbr_inputs: int,
        nbr_outputs: int,
        fct_for_x_dot: Callable[[np.ndarray, np.ndarray], np.ndarray],
        fction_for_y: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ):
        super().__init__(
            nbr_states,
            nbr_inputs,
            nbr_outputs,
            fct_for_x_dot,
            fction_for_y,
        )
        self.ran_checks_for_compute_state_derivative = False
        self.ran_checks_for_compute_output = False


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
        if not self.ran_checks_for_compute_state_derivative:
            self._check_valid_input(u)
        x_dot = self.fct_for_x_dot(x, u)

        if not self.ran_checks_for_compute_state_derivative:
            self._check_valid_state(x_dot)
            self.ran_checks_for_compute_state_derivative = True
        return x_dot

    def compute_output(self, x : np.ndarray, u: np.ndarray):
        x = self._check_valid_state(x)
        if not self.ran_checks_for_compute_output:
            self._check_valid_input(u)

        y = self.fction_for_y(x, u)
        if not self.ran_checks_for_compute_output:
            self._check_valid_output(y)
            self.ran_checks_for_compute_output = True
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

    ran_checks_for_compute_x_dot_process_model: bool
    ran_checks_for_compute_z_dot_disturb_observer: bool
    ran_checks_for_compute_state_derivative: bool
    ran_checks_for_compute_output: bool

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

        self.ran_checks_for_compute_x_dot_process_model: bool = False
        self.ran_checks_for_compute_z_dot_disturb_observer: bool = False
        self.ran_checks_for_compute_state_derivative: bool = False
        self.ran_checks_for_compute_output: bool = False

    def _check_valid_state(self, x: np.ndarray):
        if x.shape != (self.nbr_states, 1):
            raise ValueError(f"Did not provide correct number of states. System has {self.nbr_states} states, {x.shape} states were provided.")

    def _check_valid_input(self, u: np.ndarray) :
        if u.shape != (self.nbr_inputs, 1):
            raise ValueError(f"Did not provide correct number of inputs. System has {self.nbr_inputs} inputs, {u.shape} inputs were provided.")

    def _check_valid_disturbance(self, d: np.ndarray):
        if d.shape != (self.nbr_disturbances, 1):
            raise ValueError(f"Did not provide correct number of disturbances. System has {self.nbr_disturbances} disturbances, {d.shape} disturbances were provided.")
    
    def _check_valid_total_state(self, total_state: np.ndarray) -> np.ndarray:
        if total_state.shape[0] != self.nbr_total_states:
            raise ValueError(f"Did not provide correct number of states. System has {self.nbr_total_states} states, {total_state.shape} states were provided.")
        return np.reshape(total_state, (self.nbr_total_states, 1))

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
        if not self.ran_checks_for_compute_x_dot_process_model:
            self._check_valid_state(x)
            self._check_valid_input(u)
            self._check_valid_disturbance(d)
            self._check_valid_disturbance(d_estim)

        g_2 = self.g_2(x)

        x_dot = self.f(x) + self.g_1(x) @ u + g_2 @ d - g_2 @ d_estim
        
        if not self.ran_checks_for_compute_x_dot_process_model:
            self._check_valid_state(x_dot)
            self.ran_checks_for_compute_x_dot_process_model = True

        return x_dot

    def compute_z_dot_disturb_observer(self, x: np.ndarray, z:np.ndarray, u: np.ndarray):
        if not self.ran_checks_for_compute_z_dot_disturb_observer:
            self._check_valid_state(x)
            self._check_valid_disturbance(z)
            self._check_valid_input(u)

        L_d = self.L_d(x)
        g_2 = self.g_2(x)
        g_1 = self.g_1(x)
        f = self.f(x)
        p = self.p(x)

        g_1_u = g_1 @ u

        g_2_p = g_2 @ p

        g_2_z = g_2 @ z

        L_d_g_2_z = L_d @ g_2_z 

        z_dot = -L_d_g_2_z - L_d @ ( f + g_1_u + g_2_p)

        if not self.ran_checks_for_compute_z_dot_disturb_observer:
            self._check_valid_disturbance(z_dot)
            self.ran_checks_for_compute_z_dot_disturb_observer = True

        return z_dot

    def compute_state_derivative(self, total_state: np.ndarray, total_input: np.ndarray) -> np.ndarray:
        total_state = self._check_valid_total_state(total_state)
        if not self.ran_checks_for_compute_state_derivative:
            self._check_valid_total_input(total_input)
        
        x = total_state[0:self.nbr_states, :]
        z = total_state[self.nbr_states:self.nbr_states+self.nbr_disturbances, :]

        u = total_input[0:self.nbr_inputs, :]
        d = total_input[self.nbr_inputs:self.nbr_inputs+self.nbr_disturbances, :]

        z_dot = self.compute_z_dot_disturb_observer(x, z, u)

        d_estim = z + self.p(x)

        x_dot = self.compute_x_dot_process_model(x, u, d, d_estim)

        total_state_dot = np.block([
            [x_dot],
            [z_dot]
        ])
        if not self.ran_checks_for_compute_state_derivative:
            total_state_dot = self._check_valid_total_state(total_state_dot)
            self.ran_checks_for_compute_state_derivative = True

        return total_state_dot

    def compute_output(self, total_state: np.ndarray, total_input: np.ndarray) -> np.ndarray:
        total_state = self._check_valid_total_state(total_state)
        
        x = total_state[0:self.nbr_states, :]
        z = total_state[self.nbr_states:self.nbr_states+self.nbr_disturbances, :]

        d_estim = z + self.p(x)

        total_output = np.block([
            [x],
            [d_estim]
        ])

        if not self.ran_checks_for_compute_output:
            self._check_valid_total_output(total_output)
            self.ran_checks_for_compute_output = True

        return total_output

    def generate_process_model(self) -> ProcessModelToIntegrate:
        return ProcessModelToIntegrate(self.nbr_total_states, self.nbr_total_inputs, self.nbr_total_outputs, self.compute_state_derivative, self.compute_output)


def integrate_dopm(dopmg: DisturbanceObservedProcessModelGenerator, fction_for_input_at_t: Callable[[float], np.ndarray]):
    pm2i = dopmg.generate_process_model()
    fction_to_integrate = pm2i.generate_fction_to_integrate(fction_for_input_at_t)

    x0 = np.zeros((pm2i.nbr_states, 1)).ravel()

    dt = 1e-3
    t_start = 0
    t_end = 100
    t = np.arange(t_start, t_end, dt)
    # Find time-domain response by integrating the ODE
    sol = integrate.solve_ivp(
        fction_to_integrate,
        (t_start, t_end),
        x0,
        t_eval=t,
        rtol=1e-6,
        atol=1e-6,
        method='RK45',
    )

    sol_x = sol.y
    sol_t = sol.t

    sol_u, sol_y = pm2i.recreate_input_and_output(compute_input_and_disturb_from_t, sol_t, sol_x)

    fig, ax = plt.subplots(sol_y.shape[0],1)

    for i in range(dopmg.nbr_states):
        ax[i].set_xlabel(r'$t$ (s)')
        ax[i].set_ylabel(r'$q(t)$ (N)')
        # Plot data

        ax[i].plot(sol_t, sol_y[i,:], label=r'$q(t)$', color='C0')
        ax[i].plot(sol_t, sol_u[0,:], label=r'$u_{in}(t)$', color='C1')
        ax[i].legend(loc='upper right')


    for i in range(dopmg.nbr_disturbances):
        j = i + dopmg.nbr_states
        ax[j].set_xlabel(r'$t$ (s)')
        ax[j].set_ylabel(r'$d(t)$ (N)')
        # Plot data

        ax[j].plot(sol_t, sol_y[j,:], label=r'$d_{estimate}(t)$', color='C0')
        ax[j].plot(sol_t, sol_u[i+1,:], label=r'$d(t)$', color='C1')
        ax[j].legend(loc='upper right')

    fig.tight_layout()

if __name__ == '__main__':
    k_1 = 0.9
    k_2 = 0.02
    c = 0.2  # Ns/m
    m = 1 # kg

    # observer parameters

    alpha_d = 2.3
    K1_d = 0
    K2_d = 0


    def f(x:np.ndarray) -> np.ndarray:
        x1 = x[0,0]
        x2 = x[1,0]

        return np.array([
            [x2],
            [f.cnst_0 * x2 + f.cnst_1 * x1 + f.cnst_2 * x1*x1 ]
        ])

    f.cnst_0 = -c/m
    f.cnst_1 = -k_1/m
    f.cnst_2 = -k_2/m

    def g_1(x:np.ndarray) -> np.ndarray:
        return g_1.cnst_0

    g_1.cnst_0  = np.array([
                    [0],
                    [1/m]
                ])

    def g_2(x:np.ndarray) -> np.ndarray:
        return g_2.cnst_0

    g_2.cnst_0  = np.array([
                    [1, 0],
                    [0, 1/m]
                ])

    def L_d(x:np.ndarray) -> np.ndarray:
        return L_d.cnst_0

    L_d.cnst_0 = np.array([
        [alpha_d, 0],
        [0, m*alpha_d]
    ])

    def p(x:np.ndarray) -> np.ndarray:
        x1 = x[1,0]
        x2 = x[1,0]
        return np.array([
            [p.cnst_0 * x1 + p.cnst_1],
            [p.cnst_2 * x2 + p.cnst_3]
        ])

    p.cnst_0 = alpha_d
    p.cnst_1 = K1_d
    p.cnst_2 = m*alpha_d
    p.cnst_3 = K2_d

    def give_square_value(t, period, half_period, amplitude):
        t_remainder = t

        while (t_remainder >= period):
            t_remainder -= period

        return amplitude if t_remainder >= half_period else -amplitude


    def compute_input_and_disturb_from_t(t: float) -> np.ndarray:
        # input is a step
        am_u_in = 5
        if t > 0.01:
            u_in = am_u_in # N
        else:
            u_in = 0 # N
        
        
        # Sine Disturbance
        # d = compute_input_and_disturb_from_t.amplitude_d_1 * np.sin(2*np.pi*t/compute_input_and_disturb_from_t.period_d_1)

        # Square Disturbance
        

        d_1 = give_square_value(
            t, 
            compute_input_and_disturb_from_t.period_d_1, 
            compute_input_and_disturb_from_t.half_period_d_1, 
            compute_input_and_disturb_from_t.amplitude_d_1
        )

        d_2 = give_square_value(
            t, 
            compute_input_and_disturb_from_t.period_d_2, 
            compute_input_and_disturb_from_t.half_period_d_2, 
            compute_input_and_disturb_from_t.amplitude_d_2
        )


        return np.array([
            [u_in],
            [d_1],
            [d_2]
        ])

    compute_input_and_disturb_from_t.amplitude_d_1 = .3
    freq_d_1 = 0.02
    compute_input_and_disturb_from_t.period_d_1 = 1 / freq_d_1
    compute_input_and_disturb_from_t.half_period_d_1 = compute_input_and_disturb_from_t.period_d_1 / 2

    compute_input_and_disturb_from_t.amplitude_d_2 = .02
    freq_d_2 = 0.3
    compute_input_and_disturb_from_t.period_d_2 = 1 / freq_d_2
    compute_input_and_disturb_from_t.half_period_d_2 = compute_input_and_disturb_from_t.period_d_2 / 2

    dopmg = DisturbanceObservedProcessModelGenerator(2, 1, 2, f, g_1, g_2, L_d, p)

    integrate_dopm(dopmg, compute_input_and_disturb_from_t)

    plt.show()
    



    



