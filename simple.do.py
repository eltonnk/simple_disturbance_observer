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
            raise ValueError("Did not provide correct number of states.")
        return np.reshape(x, (self.nbr_states, 1))

    def _check_valid_input(self, u: np.ndarray) :
        if u.shape != (self.nbr_inputs, 1):
            raise ValueError("Did not provide correct number of inputs.")
            
    def _check_valid_output(self, y: np.ndarray):
        if y.shape != (self.nbr_outputs, 1):
            raise ValueError("Did not produce correct number of outputs.")
    
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



if __name__ == '__main__':

    # np.seterr(all='warn')
    

    k = 20.  # kg
    c = .2  # Ns/m
    m = 0.1 # kg
    A_n = np.array([
        [0.,1.],
        [-k/m, -c/m]
    ])
    B_n = np.array([
        [0.],
        [1/m]
    ])

    C_n = np.array([
        [1.0, 0.0]
    ])
    D_n = np.array([
        [0.]
    ])

    low_pass_cutoff_frequency = 500
    low_pass_damping_coeff = 0.707

    ss_n = control.StateSpace(A_n, B_n, C_n, D_n)

    P_n = control.ss2tf(ss_n) # this is the transfer fucntion from input force to ouput position

    s = control.tf('s')
    
    Q = low_pass_cutoff_frequency**2 /(s**2 + 2*low_pass_damping_coeff*low_pass_cutoff_frequency*s + low_pass_cutoff_frequency**2)

    Q_P_inverse = control.minreal(Q / P_n)

    ss_Q = control.tf2ss(Q)

    print(control.pole(ss_Q))

    ss_Q_P_inverse = control.tf2ss(Q_P_inverse)

    print(control.pole(ss_Q_P_inverse))

    A_q = ss_Q.A
    B_q = ss_Q.B
    C_q = ss_Q.C
    D_q = ss_Q.D

    A_c = ss_Q_P_inverse.A
    B_c = ss_Q_P_inverse.B
    C_c = ss_Q_P_inverse.C
    D_c = ss_Q_P_inverse.D

    M_inv = np.linalg.inv(np.eye(D_q.shape[0]) - D_q + D_c @ D_n)

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
    I_m_M_inv_D_c_Dn = np.eye(M_inv_D_c_Dn.shape[0]) - M_inv_D_c_Dn
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

    print(control.pole(ss_tot))

    ss2i_tot = StateSpaceToIntegrate(ss_tot)


    def compute_input_and_disturb_from_t(t: float) -> np.ndarray:
        # input is a step
        am_u_in = 5
        if t > 0.01:
            u_in = am_u_in # N
        else:
            u_in = 0 # N
        
        amp_d = 0.5 
        d = amp_d * np.sin(t)

        return np.array([
            [u_in],
            [d]
        ])

    fction_to_integrate = ss2i_tot.generate_fction_to_integrate(compute_input_and_disturb_from_t)

    x0 = np.zeros((A_tot.shape[0], 1)).ravel()


    dt = 1e-3
    t_start = 0
    t_end = 5
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

    fig, ax = plt.subplots(2,1)

    ax[0].set_xlabel(r'$t$ (s)')
    ax[0].set_ylabel(r'$q(t)$ (N)')
    # Plot data

    ax[0].plot(sol_t, sol_y[0,:], label=r'$q(t)$', color='C0')
    ax[0].plot(sol_t, sol_u[0,:], label=r'$u_{in}(t)$', color='C1')
    ax[0].legend(loc='upper right')

    ax[1].set_xlabel(r'$t$ (s)')
    ax[1].set_ylabel(r'$d(t)$ (N)')
    # Plot data

    ax[0].plot(sol_t, sol_y[1,:], label=r'$d(t)$', color='C0')
    ax[0].plot(sol_t, sol_u[1,:], label=r'$d(t)$', color='C1')
    # ax[1].legend(loc='upper right')

    fig.tight_layout()
    plt.show()











    