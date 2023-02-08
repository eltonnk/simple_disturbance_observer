from typing import Callable

import numpy as np
import control
import scipy.integrate as integrate
import matplotlib.pyplot as plt

import warnings

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


def list2d_tf_2_ss(list2d_tf: list[list[control.TransferFunction]]) -> control.StateSpace:
    """Takes a 2D list of SISO transfer functions that share the same 
    denominator to generate the state space representation of a MIMO system.
    Parameters
    ----------
    list2d_tf : list[list[control.TransferFunction]]
        Each row of this 2D list represents one output of the MIMO system.
        Each column of this 2D list represents one input of the MIMO system.

    Returns
    -------
    control.StateSpace
        Time domain representation of the MIMO system.
    """
    
    num=[]
    den=[]
    for list1d_tf in list2d_tf:
        num.append([])
        den.append([])
        for tf in list1d_tf:
            num[-1].append(tf.num[0][0])
            den[-1].append(tf.den[0][0])
    return control.tf2ss(num, den)

if __name__ == '__main__':

    # np.seterr(all='warn')
    

    k = 0.9
    c = 0.2  # Ns/m
    m = 1 # kg
    A = np.array([
        [0.,1.],
        [-k/m, -c/m]
    ])
    B = np.array([
        [0.],
        [1/m]
    ])

    C = np.array([
        [1.0, 0.0]
    ])
    D = np.array([
        [0.]
    ])

    low_pass_cutoff_frequency = 1000
    low_pass_damping_coeff = 0.707
    low_pass_cutoff_frequency_2 = 1000

    ss = control.StateSpace(A, B, C, D)

    # P is the real tf, P_n is the dientified tf

    P = control.ss2tf(ss) # this is the transfer fucntion from input force to output position
    P_n = P * 1 # for now they are both the same plant 

    s = control.tf('s')
    
    Q = low_pass_cutoff_frequency**2 /(s**2 + 2*low_pass_damping_coeff*low_pass_cutoff_frequency*s + low_pass_cutoff_frequency**2)
    Q = Q * low_pass_cutoff_frequency_2/(s + low_pass_cutoff_frequency_2)

    denom = (Q * (P - P_n) + P_n)
    tf_u_in2y = P * P_n / denom
    tf_d2y = P * P_n * ( 1- Q ) / denom
    

    tf_tot = ([
        [tf_u_in2y,tf_d2y]
    ])



    ss_tot = list2d_tf_2_ss(tf_tot)

    print(control.pole(ss_tot))

    ss2i_tot = StateSpaceToIntegrate(ss_tot)

    def compute_input_and_disturb_from_t(t: float) -> np.ndarray:
        # input is a step
        am_u_in = 5
        if t > 0.01:
            u_in = am_u_in # N
        else:
            u_in = 0 # N
        
        amp_d = 0.1
        freq_d = 50
        d = amp_d * np.sin(2*np.pi*freq_d*t)

        return np.array([
            [u_in],
            [d]
        ])

    fction_to_integrate = ss2i_tot.generate_fction_to_integrate(compute_input_and_disturb_from_t)

    x0 = np.zeros((ss_tot.A.shape[0], 1)).ravel()


    dt = 1e-3
    t_start = 0
    t_end = 70
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

    sol_u, sol_y = ss2i_tot.recreate_input_and_output(compute_input_and_disturb_from_t, sol_t, sol_x)

    fig, ax = plt.subplots(1,1)
    ax.set_xlabel(r'$t$ (s)')
    ax.set_ylabel(r'$q(t)$ (N)')
    # Plot data

    ax.plot(sol_t, sol_y[0,:], label=r'$q(t)$', color='C0')
    ax.plot(sol_t, sol_u[0,:], label=r'$u_{in}(t)$', color='C1')
    ax.legend(loc='upper right')

    # ax[0].set_xlabel(r'$t$ (s)')
    # ax[0].set_ylabel(r'$q(t)$ (N)')
    # # Plot data

    # ax[0].plot(sol_t, sol_y[0,:], label=r'$q(t)$', color='C0')
    # ax[0].plot(sol_t, sol_u[0,:], label=r'$u_{in}(t)$', color='C1')
    # ax[0].legend(loc='upper right')

    # ax[1].set_xlabel(r'$t$ (s)')
    # ax[1].set_ylabel(r'$d(t)$ (N)')
    # # Plot data

    # ax[1].plot(sol_t, sol_y[1,:], label=r'$d_{estimate}(t)$', color='C0')
    # ax[1].plot(sol_t, sol_u[1,:], label=r'$d(t)$', color='C1')
    # ax[1].legend(loc='upper right')

    fig.tight_layout()
    plt.show()











    