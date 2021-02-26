from __future__ import division

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from pylab import figure
from scipy.sparse.linalg import gmres


class prop_rock(object):
    """
    This is a class that captures rock physical properties, including permeability, porosity, and
    compressibility.
    """
    def __init__(self, kx=0, ky=0, por=0, cr=0, kro=0, dkro=0, krg=0, dkrg=0):
        self.kx = kx
        self.ky = ky
        self.por = por
        self.cr = cr
        self.kro = kro
        self.krg = krg
        self.dkro = dkro
        self.dkrg = dkrg

    def calc_kro(self, sg):
        self.kro = (1 - sg) ** 1.5
        return self.kro

    def calc_dkro(self, sg):
        self.dkro = -1.5 * (1 - sg) ** 0.5
        return self.dkro

    def calc_krg(self, sg):
        self.krg = (sg) ** 2
        return self.krg

    def calc_dkrg(self, sg):
        self.dkrg = 2 * sg
        return self.dkrg

    def plot_kro(self):
        sgx = np.linspace(0, 1, 500)
        kro_try = []
        dkro_try = []
        for i in sgx:
            kro_try.append(prop_rock.calc_kro(self, i))
            dkro_try.append(prop_rock.calc_dkro(self, i))
        f, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(sgx, kro_try, 'r-', label=r'$kr_o$')
        ax2.plot(sgx, dkro_try, 'r--', label=r'$\frac{\partial kro_o}{\partial sg}$')
        ax1.set_xlabel('Gas Saturation (fraction)')
        ax1.set_ylabel('kro')
        ax2.set_ylabel('kro derivative')
        ax1.legend(loc=2)
        ax2.legend(loc=1)
        plt.grid()
        plt.show()
        return f

    def plot_krg(self):
        sgx = np.linspace(0, 1, 500)
        krg_try = []
        dkrg_try = []
        for i in sgx:
            krg_try.append(prop_rock.calc_krg(self, i))
            dkrg_try.append(prop_rock.calc_dkrg(self, i))
        f, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(sgx, krg_try, 'r-', label=r'$kr_g$')
        ax2.plot(sgx, dkrg_try, 'r--', label=r'$\frac{\partial kr_g}{\partial sg}$')
        ax1.set_xlabel('Gas Saturation (fraction)')
        ax1.set_ylabel('krg')
        ax2.set_ylabel('krg derivative')
        ax1.legend(loc=2)
        ax2.legend(loc=1)
        plt.grid()
        plt.show()
        return f

    def plot_all(self):
        f1 = prop_rock.plot_kro(self)
        f2 = prop_rock.plot_krg(self)
        return f1, f2


class prop_fluid(object):
    """
    This object contains fluid properties. Phase: oil and gas. Isothermal; only a function of pressure.
    """
    def __init__(self, c_o=0, mu_o=0, rho_o=0, mu_g=0, dmu_g=0, rmu_g=0, p_bub=0, p_atm=14.7, b_o=0, b_g=0,
                 dp=0, rs=0, db_o=0, db_g=0, drs=0):
        self.c_o = c_o
        self.mu_o = mu_o
        self.rho_o = rho_o
        self.mu_g = mu_g
        self.rmu_g = rmu_g
        self.dmu_g = dmu_g
        self.p_bub = p_bub
        self.p_atm = p_atm
        self.b_o = b_o
        self.db_o = db_o
        self.b_g = b_g
        self.db_g = db_g
        self.dp = dp
        self.rs = rs
        self.drs = drs

    def calc_mu_g(self, p):
        self.mu_g = 3e-10 * p ** 2 + 1e-6 * p + 0.0133
        return self.mu_g

    def calc_dmu_g(self, p):
        self.dmu_g = 3e-10 * 2 * p + 1e-6
        return self.dmu_g

    def calc_rmu_g(self, p):
        self.rmu_g = 20000000000 * (3 * p + 5000) / (3 * p ** 2 + 10000 * p + 133000000) ** 2
        return self.rmu_g

    def calc_dp(self, p):
        if p < self.p_bub:
            self.dp = self.p_atm - p
        else:
            self.dp = self.p_atm - self.p_bub
        return self.dp

    def calc_bo(self, p):
        if p < self.p_bub:
            self.b_o = 1 / np.exp(-8e-5 * (self.p_atm - p))
        else:
            self.b_o = 1 / (np.exp(-8e-5 * (self.p_atm - self.p_bub)) * np.exp(-self.c_o * (p - self.p_bub)))
        return self.b_o

    def calc_dbo(self, p):
        if p < self.p_bub:
            self.db_o = -8e-5 * np.exp(8e-5 * (self.p_atm - p))
        else:
            self.db_o = self.c_o * np.exp(8e-5 * (self.p_atm - self.p_bub)) * np.exp(self.c_o * (p - self.p_bub))
        return self.db_o

    def calc_bg(self, p):
        self.b_g = 1 / (np.exp(1.7e-3 * prop_fluid.calc_dp(self, p)))
        return self.b_g

    def calc_dbg(self, p):
        if p < self.p_bub:
            self.db_g = 1.7e-3 * np.exp(-1.7e-3 * prop_fluid.calc_dp(self, p))
        else:
            self.db_g = 0
        return self.db_g

    def calc_rs(self, p):
        if p < self.p_bub:
            rs_factor = 1
        else:
            rs_factor = 0
        self.rs = 178.11 ** 2 / 5.615 * ((p / self.p_bub) ** 1.3 * rs_factor + (1 - rs_factor))
        return self.rs

    def calc_drs(self, p):
        if p < self.p_bub:
            rs_factor = 1
        else:
            rs_factor = 0
        self.drs = 178.11 ** 2 / 5.615 * (1.3 * p ** 0.3 / self.p_bub ** 1.3 * rs_factor + 0 * (1 - rs_factor))
        return self.drs

    def plot_bo(self):
        px = np.linspace(1, 5000, 1000)
        bo_try = []
        dbo_try = []
        for i in px:
            bo_try.append(prop_fluid.calc_bo(self, i))
            dbo_try.append(prop_fluid.calc_dbo(self, i))
        f, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(px, bo_try, 'r-', label=r'$b_o$')
        ax2.plot(px, dbo_try, 'r--', label=r'$\frac{\partial b_o}{\partial p}$')
        ax1.set_xlabel('Pressure (psi)')
        ax1.set_ylabel('Oil Shrinkage (RB/STB)')
        ax2.set_ylabel('Oil Shrinkage Derivative')
        ax1.legend(loc=2)
        ax2.legend(loc=1)
        plt.grid()
        plt.show()
        return f

    def plot_bg(self):
        px = np.linspace(1, 5000, 1000)
        bg_try = []
        dbg_try = []
        for i in px:
            bg_try.append(prop_fluid.calc_bg(self, i))
            dbg_try.append(prop_fluid.calc_dbg(self, i))
        f, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(px, bg_try, 'r-', label=r'$b_g$')
        ax2.plot(px, dbg_try, 'r--', label=r'$\frac{\partial b_g}{\partial p}$')
        ax1.set_xlabel('Pressure (psi)')
        ax1.set_ylabel('Gas Shrinkage (RB/STB)')
        ax2.set_ylabel('Gas Shrinkage Derivative')
        ax1.legend(loc=2)
        ax2.legend(loc=1)
        plt.grid()
        plt.show()
        return f

    def plot_rs(self):
        px = np.linspace(1, 5000, 1000)
        rs_try = []
        drs_try = []
        for i in px:
            rs_try.append(prop_fluid.calc_rs(self, i))
            drs_try.append(prop_fluid.calc_drs(self, i))
        f, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(px, rs_try, 'r-', label=r'$R_s$')
        ax2.plot(px, drs_try, 'r--', label=r'$\frac{\partial R_s}{\partial p}$')
        ax1.set_xlabel('Pressure (psi)')
        ax1.set_ylabel('Solution Gas-Oil Ratio (RB/STB)')
        ax2.set_ylabel('Solution Gas-Oil Ratio Derivative')
        ax1.legend(loc=2)
        ax2.legend(loc=1)
        plt.grid()
        plt.show()
        return f

    def plot_mu_g(self):
        px = np.linspace(1, 5000, 1000)
        mu_g_try = []
        dmu_g_try = []
        for i in px:
            mu_g_try.append(prop_fluid.calc_mu_g(self, i))
            dmu_g_try.append(prop_fluid.calc_dmu_g(self, i))
        f, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(px, mu_g_try, 'r-', label=r'$\mu_g$')
        ax2.plot(px, dmu_g_try, 'r--', label=r'$\frac{\partial \mu_g}{\partial p}$')
        ax1.set_xlabel('Pressure (psi)')
        ax1.set_ylabel('Gas Viscosity (cp)')
        ax2.set_ylabel('Gas Viscosity Derivative')
        ax1.legend(loc=2)
        ax2.legend(loc=1)
        plt.grid()
        plt.show()
        return f

    def plot_all(self):
        f1 = prop_fluid.plot_bo(self)
        f2 = prop_fluid.plot_bg(self)
        f3 = prop_fluid.plot_rs(self)
        f4 = prop_fluid.plot_mu_g(self)
        return f1, f2, f3, f4


class prop_grid(object):
    """This describes grid dimension and numbers."""
    def __init__(self, Nx=0, Ny=0, Nz=0, dx=0, dy=0, dz=0):
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.dx = dx
        self.dy = dy
        self.dz = dz

    def grid_dimension_x(self, Lx):
        self.dx = Lx / self.Nx
        return self.dx

    def grid_dimension_y(self, Ly):
        self.dy = Ly / self.Ny
        return self.dy

    def grid_dimension_z(self, Lz):
        self.dz = Lz / self.Nz
        return self.dz


class prop_res(object):
    """A class that captures reservoir dimension and initial pressure."""
    def __init__(self, Lx=0, Ly=0, Lz=0, press_n=0, sg_n=0, press_n1_k=0, sg_n1_k=0):
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.press_n = press_n
        self.sg_n = sg_n
        self.press_n1_k = press_n1_k
        self.sg_n1_k = sg_n1_k

    def stack_ps(self, press_n, sg_n):
        stacked_vector = []
        for i in range(len(press_n)):
            stacked_vector.append(press_n[i])
            stacked_vector.append(sg_n[i])
        return np.asarray(stacked_vector)


class prop_well(object):
    """Describes well location a flow rate. Also provides conversion from
    cartesian i,j coordinate to grid number"""
    def __init__(self, loc=0, q=0, q_lim=0, pwf=0, pwf_lim=0, rw=0, qo_control=True):
        self.loc = loc
        self.pwf = pwf
        self.q = q
        self.rw = rw
        self.qo_control = qo_control
        self.q_lim = q_lim
        self.pwf_lim = pwf_lim

    def index_to_grid(self, Nx):
        return (self.loc[1] - 1) * Nx + self.loc[0]


class prop_time(object):
    """Describes time-step (assumed constant) and time interval"""
    def __init__(self, tstep=0, tmax=0):
        self.tstep = tstep
        self.tmax = tmax


def load_data(filename):
    """Loads ECLIPSE simulation block pressure data as a comparison"""
    df = pd.read_csv(filename)

    t1_ecl = df.loc[:, ['TIME']]  # Time in simulation: DAY
    pwf_ecl = df.loc[:, ['WBHP:PWELL01']]
    bpr_ecl = df.loc[:, ['BPR:(12,12,1)']]
    fpr_ecl = df.loc[:, ['FPR']]
    qo1_ecl = df.loc[:, ['FOPR']]
    qg1_ecl = df.loc[:, ['FGPR']]
    return t1_ecl, pwf_ecl, bpr_ecl, fpr_ecl, qo1_ecl, qg1_ecl


def calc_transmissibility_x(k_x, kr_o, mu_o, b_o, params, i, j):
    """Calculates transmissibility in x-direction"""
    # Calculate transmissibility in x-direction. Unit: (md ft psi)/cp
    dx = params['dx']
    dy = params['dy']
    dz = params['dz']
    p_grids = params['p_grids_n1']

    # Arithmetic Average for k
    k_x_avg = (dx[j, i] + dx[j, i + 1]) / (dx[j, i] / k_x[j, i] + dx[j, i + 1] / k_x[j, i + 1])
    A = dy[j, i] * dz[j, i]
    x_l = (dx[j, i] + dx[j, i + 1]) / 2

    fluid_term = upwind(p_grids, [kr_o, b_o, 1 / mu_o], i, j, dir='x')
    return k_x_avg * A / x_l * fluid_term


def calc_transmissibility_y(k_y, kr_o, mu_o, b_o, params, i, j):
    """Calculates transmissibility in y-direction"""
    # Calculate transmissibility in y-direction. Unit: (md ft psi)/cp
    dx = params['dx']
    dy = params['dy']
    dz = params['dz']
    p_grids = params['p_grids_n1']

    # Arithmetic Average for k
    k_x_avg = (dy[j, i] + dy[j + 1, i]) / (dy[j, i] / k_y[j, i] + dy[j + 1, i] / k_y[j + 1, i])
    A = dx[j, i] * dz[j, i]
    x_l = (dy[j, i] + dy[j + 1, i]) / 2

    fluid_term = upwind(p_grids, [kr_o, b_o, 1 / mu_o], i, j, dir='y')
    return k_x_avg * A / x_l * fluid_term


def upwind(p_grids, pars, i, j, dir):
    # upwind parameters based on pressure values between two blocks
    if dir == 'x':
        if p_grids[j, i] > p_grids[j, i + 1]:
            mult = 1
            for p in range(len(pars)):
                mult *= pars[p][j, i]
        else:
            mult = 1
            for p in range(len(pars)):
                mult *= pars[p][j, i + 1]
    elif dir == 'y':
        if p_grids[j, i] > p_grids[j + 1, i]:
            mult = 1
            for p in range(len(pars)):
                mult *= pars[p][j, i]
        else:
            mult = 1
            for p in range(len(pars)):
                mult *= pars[p][j + 1, i]
    return mult


def ij_to_grid(i, j, Nx):
    # Convert i,j coordinate to block number
    return (i) + Nx * j


def flip_variables(M, ind):
    # Flip variables (for Residual and Jacobian). Ind==0 for vector, 1 for 2D matrix.
    J = M * 1
    m = J.shape[0]
    for i in range(m):
        if i % 2 == 0:
            if ind == 0:
                J[i:i + 2] = np.flip(J[i:i + 2], 0)
            elif ind == 1:
                J[i:i + 2, :] = np.flip(J[i:i + 2, :], 0)
            else:
                print('Unknown Choice..')
    return J


def construct_T(mat, params):
    # Create matrix T containing connection transmissibilities of all blocks
    k_x = params['k_x']
    k_y = params['k_y']
    b_o = params['b_o']
    b_g = params['b_g']
    mu_o = params['mu_o']
    mu_g = params['mu_g']
    kr_o = params['kr_o']
    kr_g = params['kr_g']
    rs = params['rs']
    p_grids = params['p_grids_n1']

    m = mat.shape[0]
    n = mat.shape[1]
    T = np.zeros((m * n * 2, m * n * 2))
    for j in range(m):
        for i in range(n):
            # 2 neighbors in x direction
            if i < n - 1:
                # Oil D1
                T[(mat[j, i] - 1) * 2, (mat[j, i + 1] - 1) * 2] = calc_transmissibility_x(k_x, kr_o, mu_o, b_o, params,
                                                                                          i, j)
                T[(mat[j, i + 1] - 1) * 2, (mat[j, i] - 1) * 2] = T[(mat[j, i] - 1) * 2, (mat[j, i + 1] - 1) * 2]

                # Gas D3
                T[(mat[j, i] - 1) * 2 + 1, (mat[j, i + 1] - 1) * 2] = calc_transmissibility_x(k_x, kr_g, mu_g, b_g,
                                                                                              params, i, j) + upwind(
                    p_grids, [rs], i, j, dir='x') * calc_transmissibility_x(k_x, kr_o, mu_o, b_o, params, i, j)
                T[(mat[j, i + 1] - 1) * 2 + 1, (mat[j, i] - 1) * 2] = T[
                    (mat[j, i] - 1) * 2 + 1, (mat[j, i + 1] - 1) * 2]
            # 2 neighbors in y direction
            if j < m - 1:
                # Oil D1
                T[(mat[j, i] - 1) * 2, (mat[j + 1, i] - 1) * 2] = calc_transmissibility_y(k_y, kr_o, mu_o, b_o, params,
                                                                                          i, j)
                T[(mat[j + 1, i] - 1) * 2, (mat[j, i] - 1) * 2] = T[(mat[j, i] - 1) * 2, (mat[j + 1, i] - 1) * 2]

                # Gas D3
                T[(mat[j, i] - 1) * 2 + 1, (mat[j + 1, i] - 1) * 2] = calc_transmissibility_y(k_y, kr_g, mu_g, b_g,
                                                                                              params, i, j) + upwind(
                    p_grids, [rs], i, j, dir='y') * calc_transmissibility_y(k_y, kr_o, mu_o, b_o, params, i, j)
                T[(mat[j + 1, i] - 1) * 2 + 1, (mat[j, i] - 1) * 2] = T[
                    (mat[j, i] - 1) * 2 + 1, (mat[j + 1, i] - 1) * 2]

    for k in range(T.shape[0]):
        # For 2 phases only, not generalized to n phases
        if k % 2 == 0:
            T[k, k] = -np.sum(T[k, ::2])
            T[k, k + 1] = -np.sum(T[k, 1::2])
        else:
            T[k, k - 1] = -np.sum(T[k, ::2])
            T[k, k] = -np.sum(T[k, 1::2])
    T = T * 0.001127
    return T


def construct_J(mat, params, props, T):
    # Construct Jacobian matrix
    k_x = params['k_x']
    k_y = params['k_y']
    b_o = params['b_o']
    db_o = params['db_o']
    b_g = params['b_g']
    db_g = params['db_g']
    mu_o = params['mu_o']
    mu_g = params['mu_g']
    dmu_g = params['dmu_g']
    kr_o = params['kr_o']
    dkr_o = params['dkr_o']
    kr_g = params['kr_g']
    dkr_g = params['dkr_g']
    rs = params['rs']
    drs = params['drs']
    p_grids_n1 = params['p_grids_n1']
    dx = params['dx']
    dy = params['dy']
    dz = params['dz']
    sg_n1 = params['sg_n1']
    por = props['rock'].por

    m = mat.shape[0]
    n = mat.shape[1]
    J = np.zeros((m * n * 2, m * n * 2))
    for j in range(m):
        for i in range(n):
            C1_i_neg = 0
            C1_i_pos = 0
            C1_j_neg = 0
            C1_j_pos = 0
            C2_i_neg = 0
            C2_i_pos = 0
            C2_j_neg = 0
            C2_j_pos = 0
            C3_i_neg = 0
            C3_i_pos = 0
            C3_j_neg = 0
            C3_j_pos = 0
            C4_i_neg = 0
            C4_i_pos = 0
            C4_j_neg = 0
            C4_j_pos = 0
            ## 2 neighbors in x direction
            # Right block (i+1/2) elements
            if i < n - 1:
                # Oil D1 derivative w.r.t. pressure
                dp_i_pos = (p_grids_n1[j, i + 1] - p_grids_n1[j, i])
                if p_grids_n1[j, i] < p_grids_n1[j, i + 1]:
                    D1_i_pos = dp_i_pos * calc_transmissibility_x(k_x, kr_o, mu_o, db_o, params, i, j)
                    D2_i_pos = dp_i_pos * calc_transmissibility_x(k_x, dkr_o, mu_o, b_o, params, i, j)
                    D3_i_pos_free = dp_i_pos * calc_transmissibility_x(k_x, kr_g, mu_g ** 2,
                                                                       (db_g * mu_g - b_g * dmu_g), params, i, j)
                    D3_i_pos_sol = dp_i_pos * calc_transmissibility_x(k_x, kr_o, mu_o, drs * b_o + db_o * rs, params, i,
                                                                      j)
                    D3_i_pos = D3_i_pos_free + D3_i_pos_sol
                    D4_i_pos = dp_i_pos * (
                                calc_transmissibility_x(k_x, dkr_g, mu_g, b_g, params, i, j) + calc_transmissibility_x(
                            k_x, dkr_o, mu_o, rs * b_o, params, i, j))
                    C1_i_pos = 0
                    C2_i_pos = 0
                    C3_i_pos = 0
                    C4_i_pos = 0
                else:
                    D1_i_pos = 0
                    D2_i_pos = 0
                    D3_i_pos = 0
                    D4_i_pos = 0
                    C1_i_pos = dp_i_pos * calc_transmissibility_x(k_x, kr_o, mu_o, db_o, params, i, j)
                    C2_i_pos = dp_i_pos * calc_transmissibility_x(k_x, dkr_o, mu_o, b_o, params, i, j)
                    C3_i_pos_free = dp_i_pos * calc_transmissibility_x(k_x, kr_g, mu_g ** 2,
                                                                       (db_g * mu_g - b_g * dmu_g), params, i, j)
                    C3_i_pos_sol = dp_i_pos * calc_transmissibility_x(k_x, kr_o, mu_o, drs * b_o + db_o * rs, params, i,
                                                                      j)
                    C3_i_pos = C3_i_pos_free + C3_i_pos_sol
                    C4_i_pos = dp_i_pos * (
                                calc_transmissibility_x(k_x, dkr_g, mu_g, b_g, params, i, j) + calc_transmissibility_x(
                            k_x, dkr_o, mu_o, rs * b_o, params, i, j))
                J[(mat[j, i] - 1) * 2, (mat[j, i + 1] - 1) * 2] = calc_transmissibility_x(k_x, kr_o, mu_o, b_o, params,
                                                                                          i, j) + D1_i_pos

                # Oil D2 derivative w.r.t. sg
                J[(mat[j, i] - 1) * 2, (mat[j, i + 1] - 1) * 2 + 1] = D2_i_pos

                # Gas D3 derivative w.r.t. pressure
                J[(mat[j, i] - 1) * 2 + 1, (mat[j, i + 1] - 1) * 2] = calc_transmissibility_x(k_x, kr_g, mu_g, b_g,
                                                                                              params, i,
                                                                                              j) + calc_transmissibility_x(
                    k_x, kr_o, mu_o, b_o * rs, params, i, j) + D3_i_pos

                # Gas D4 derivative w.r.t. sg
                J[(mat[j, i] - 1) * 2 + 1, (mat[j, i + 1] - 1) * 2 + 1] = D4_i_pos

            # Left block (i-1/2) elements
            if i > 0:
                # Oil D1 derivative w.r.t. pressure
                dp_i_neg = (p_grids_n1[j, i - 1] - p_grids_n1[j, i])
                if p_grids_n1[j, i] < p_grids_n1[j, i - 1]:
                    D1_i_neg = dp_i_neg * calc_transmissibility_x(k_x, kr_o, mu_o, db_o, params, i - 1, j)
                    D2_i_neg = dp_i_neg * calc_transmissibility_x(k_x, dkr_o, mu_o, b_o, params, i - 1, j)
                    D3_i_neg_free = dp_i_neg * calc_transmissibility_x(k_x, kr_g, mu_g ** 2,
                                                                       (db_g * mu_g - b_g * dmu_g), params, i - 1, j)
                    D3_i_neg_sol = dp_i_neg * calc_transmissibility_x(k_x, kr_o, mu_o, drs * b_o + db_o * rs, params,
                                                                      i - 1, j)
                    D3_i_neg = D3_i_neg_free + D3_i_neg_sol
                    D4_i_neg = dp_i_neg * (calc_transmissibility_x(k_x, dkr_g, mu_g, b_g, params, i - 1,
                                                                   j) + calc_transmissibility_x(k_x, dkr_o, mu_o,
                                                                                                rs * b_o, params, i - 1,
                                                                                                j))
                    C1_i_neg = 0
                    C2_i_neg = 0
                    C3_i_neg = 0
                    C4_i_neg = 0
                else:
                    D1_i_neg = 0
                    D2_i_neg = 0
                    D3_i_neg = 0
                    D4_i_neg = 0
                    C1_i_neg = dp_i_neg * calc_transmissibility_x(k_x, kr_o, mu_o, db_o, params, i - 1, j)
                    C2_i_neg = dp_i_neg * calc_transmissibility_x(k_x, dkr_o, mu_o, b_o, params, i - 1, j)
                    C3_i_neg_free = dp_i_neg * calc_transmissibility_x(k_x, kr_g, mu_g ** 2,
                                                                       (db_g * mu_g - b_g * dmu_g), params, i - 1, j)
                    C3_i_neg_sol = dp_i_neg * calc_transmissibility_x(k_x, kr_o, mu_o, drs * b_o + db_o * rs, params,
                                                                      i - 1, j)
                    C3_i_neg = C3_i_neg_free + C3_i_neg_sol
                    C4_i_neg = dp_i_neg * (calc_transmissibility_x(k_x, dkr_g, mu_g, b_g, params, i - 1,
                                                                   j) + calc_transmissibility_x(k_x, dkr_o, mu_o,
                                                                                                rs * b_o, params, i - 1,
                                                                                                j))
                J[(mat[j, i] - 1) * 2, (mat[j, i - 1] - 1) * 2] = calc_transmissibility_x(k_x, kr_o, mu_o, b_o, params,
                                                                                          i - 1, j) + D1_i_neg

                # Oil D2 derivative w.r.t. sg
                J[(mat[j, i] - 1) * 2, (mat[j, i - 1] - 1) * 2 + 1] = D2_i_neg

                # Gas D3 derivative w.r.t. pressure
                J[(mat[j, i] - 1) * 2 + 1, (mat[j, i - 1] - 1) * 2] = calc_transmissibility_x(k_x, kr_g, mu_g, b_g,
                                                                                              params, i - 1,
                                                                                              j) + calc_transmissibility_x(
                    k_x, kr_o, mu_o, b_o * rs, params, i - 1, j) + D3_i_neg

                # Gas D4 derivative w.r.t. sg
                J[(mat[j, i] - 1) * 2 + 1, (mat[j, i - 1] - 1) * 2 + 1] = D4_i_neg

            ## 2 neighbors in y direction
            # Lower block (j+1/2) elements
            if j < m - 1:
                # Oil D1 derivative w.r.t. pressure
                dp_j_pos = (p_grids_n1[j + 1, i] - p_grids_n1[j, i])
                if p_grids_n1[j, i] < p_grids_n1[j + 1, i]:
                    D1_j_pos = dp_j_pos * calc_transmissibility_y(k_y, kr_o, mu_o, db_o, params, i, j)
                    D2_j_pos = dp_j_pos * calc_transmissibility_y(k_y, dkr_o, mu_o, b_o, params, i, j)
                    D3_j_pos_free = dp_j_pos * calc_transmissibility_y(k_y, kr_g, mu_g ** 2,
                                                                       (db_g * mu_g - b_g * dmu_g), params, i, j)
                    D3_j_pos_sol = dp_j_pos * calc_transmissibility_y(k_y, kr_o, mu_o, drs * b_o + db_o * rs, params, i,
                                                                      j)
                    D3_j_pos = D3_j_pos_free + D3_j_pos_sol
                    D4_j_pos = dp_j_pos * (
                                calc_transmissibility_y(k_y, dkr_g, mu_g, b_g, params, i, j) + calc_transmissibility_y(
                            k_y, dkr_o, mu_o, rs * b_o, params, i, j))
                    C1_j_pos = 0
                    C2_j_pos = 0
                    C3_j_pos = 0
                    C4_j_pos = 0
                else:
                    D1_j_pos = 0
                    D2_j_pos = 0
                    D3_j_pos = 0
                    D4_j_pos = 0
                    C1_j_pos = dp_j_pos * calc_transmissibility_y(k_y, kr_o, mu_o, db_o, params, i, j)
                    C2_j_pos = dp_j_pos * calc_transmissibility_y(k_y, dkr_o, mu_o, b_o, params, i, j)
                    C3_j_pos_free = dp_j_pos * calc_transmissibility_y(k_y, kr_g, mu_g ** 2,
                                                                       (db_g * mu_g - b_g * dmu_g), params, i, j)
                    C3_j_pos_sol = dp_j_pos * calc_transmissibility_y(k_y, kr_o, mu_o, drs * b_o + db_o * rs, params, i,
                                                                      j)
                    C3_j_pos = C3_j_pos_free + C3_j_pos_sol
                    C4_j_pos = dp_j_pos * (
                                calc_transmissibility_y(k_y, dkr_g, mu_g, b_g, params, i, j) + calc_transmissibility_y(
                            k_y, dkr_o, mu_o, rs * b_o, params, i, j))
                J[(mat[j, i] - 1) * 2, (mat[j + 1, i] - 1) * 2] = calc_transmissibility_y(k_y, kr_o, mu_o, b_o, params,
                                                                                          i, j) + D1_j_pos

                # Oil D2 derivative w.r.t. sg
                J[(mat[j, i] - 1) * 2, (mat[j + 1, i] - 1) * 2 + 1] = D2_j_pos

                # Gas D3 derivative w.r.t. pressure
                J[(mat[j, i] - 1) * 2 + 1, (mat[j + 1, i] - 1) * 2] = calc_transmissibility_y(k_y, kr_g, mu_g, b_g,
                                                                                              params, i,
                                                                                              j) + calc_transmissibility_y(
                    k_y, kr_o, mu_o, b_o * rs, params, i, j) + D3_j_pos

                # Gas D4 derivative w.r.t. sg
                J[(mat[j, i] - 1) * 2 + 1, (mat[j + 1, i] - 1) * 2 + 1] = D4_j_pos

            # Upper block (j-1/2) elements
            if j > 0:
                # Oil D1 j-1 element
                dp_j_neg = (p_grids_n1[j - 1, i] - p_grids_n1[j, i])
                if p_grids_n1[j, i] < p_grids_n1[j - 1, i]:
                    D1_j_neg = dp_j_neg * calc_transmissibility_y(k_y, kr_o, mu_o, db_o, params, i, j - 1)
                    D2_j_neg = dp_j_neg * calc_transmissibility_y(k_y, dkr_o, mu_o, b_o, params, i, j - 1)
                    D3_j_neg_free = dp_j_neg * calc_transmissibility_y(k_y, kr_g, mu_g ** 2,
                                                                       (db_g * mu_g - b_g * dmu_g), params, i, j - 1)
                    D3_j_neg_sol = dp_j_neg * calc_transmissibility_y(k_y, kr_o, mu_o, drs * b_o + db_o * rs, params, i,
                                                                      j - 1)
                    D3_j_neg = D3_j_neg_free + D3_j_neg_sol
                    D4_j_neg = dp_j_neg * (calc_transmissibility_y(k_y, dkr_g, mu_g, b_g, params, i,
                                                                   j - 1) + calc_transmissibility_y(k_y, dkr_o, mu_o,
                                                                                                    rs * b_o, params, i,
                                                                                                    j - 1))
                    C1_j_neg = 0
                    C2_j_neg = 0
                    C3_j_neg = 0
                    C4_j_neg = 0
                else:
                    D1_j_neg = 0
                    D2_j_neg = 0
                    D3_j_neg = 0
                    D4_j_neg = 0
                    C1_j_neg = dp_j_neg * calc_transmissibility_y(k_y, kr_o, mu_o, db_o, params, i, j - 1)
                    C2_j_neg = dp_j_neg * calc_transmissibility_y(k_y, dkr_o, mu_o, b_o, params, i, j - 1)
                    C3_j_neg_free = dp_j_neg * calc_transmissibility_y(k_y, kr_g, mu_g ** 2,
                                                                       (db_g * mu_g - b_g * dmu_g), params, i, j - 1)
                    C3_j_neg_sol = dp_j_neg * calc_transmissibility_y(k_y, kr_o, mu_o, drs * b_o + db_o * rs, params, i,
                                                                      j - 1)
                    C3_j_neg = C3_j_neg_free + C3_j_neg_sol
                    C4_j_neg = dp_j_neg * (calc_transmissibility_y(k_y, dkr_g, mu_g, b_g, params, i,
                                                                   j - 1) + calc_transmissibility_y(k_y, dkr_o, mu_o,
                                                                                                    rs * b_o, params, i,
                                                                                                    j - 1))
                J[(mat[j, i] - 1) * 2, (mat[j - 1, i] - 1) * 2] = calc_transmissibility_y(k_y, kr_o, mu_o, b_o, params,
                                                                                          i, j - 1) + D1_j_neg

                # Oil D2 derivative w.r.t. sg
                J[(mat[j, i] - 1) * 2, (mat[j - 1, i] - 1) * 2 + 1] = D2_j_neg

                # Gas D3 derivative w.r.t. pressure
                J[(mat[j, i] - 1) * 2 + 1, (mat[j - 1, i] - 1) * 2] = calc_transmissibility_y(k_y, kr_g, mu_g, b_g,
                                                                                              params, i,
                                                                                              j - 1) + calc_transmissibility_y(
                    k_y, kr_o, mu_o, b_o * rs, params, i, j - 1) + D3_j_neg

                # Gas D4 derivative w.r.t. sg
                J[(mat[j, i] - 1) * 2 + 1, (mat[j - 1, i] - 1) * 2 + 1] = D4_j_neg

            ## Main Blocks (diagonal of matrix J)
            acc_par = dx[j, i] * dy[j, i] * dz[j, i] * por / props['time'].tstep / 5.615 / 0.001127

            # Main Diagonal 1
            diag_transmissibility1 = T[(mat[j, i] - 1) * 2, (mat[j, i] - 1) * 2] / 0.001127
            acc1 = acc_par * ((1 - sg_n1[j, i]) * db_o[j, i])
            J[(mat[j, i] - 1) * 2, (
                        mat[j, i] - 1) * 2] = diag_transmissibility1 + C1_i_neg + C1_i_pos + C1_j_neg + C1_j_pos - acc1

            # Main Diagonal 2
            acc2 = -acc_par * b_o[j, i]
            J[(mat[j, i] - 1) * 2, (mat[j, i] - 1) * 2 + 1] = C2_i_neg + C2_i_pos + C2_j_neg + C2_j_pos - acc2

            # Main Diagonal 3
            diag_transmissibility3 = T[(mat[j, i] - 1) * 2 + 1, (mat[j, i] - 1) * 2] / 0.001127
            acc3 = acc_par * (
                        sg_n1[j, i] * db_g[j, i] + (1 - sg_n1[j, i]) * (db_o[j, i] * rs[j, i] + b_o[j, i] * drs[j, i]))
            J[(mat[j, i] - 1) * 2 + 1, (
                        mat[j, i] - 1) * 2] = diag_transmissibility3 + C3_i_neg + C3_i_pos + C3_j_neg + C3_j_pos - acc3

            # Main Diagonal 4
            acc4 = acc_par * (b_g[j, i] - b_o[j, i] * rs[j, i])
            J[(mat[j, i] - 1) * 2 + 1, (mat[j, i] - 1) * 2 + 1] = C4_i_neg + C4_i_pos + C4_j_neg + C4_j_pos - acc4

    J = J * 0.001127
    return J


def construct_D(mat, params, props):
    # Construct accumulation matrix for every block
    b_o = params['b_o']
    b_g = params['b_g']
    rs = params['rs']
    p_grids_n = params['p_grids_n']
    sg_n = params['sg_n']
    sg_n1 = params['sg_n1']
    dx = params['dx']
    dy = params['dy']
    dz = params['dz']
    por = props['rock'].por

    m = mat.shape[0]
    n = mat.shape[1]

    D = np.zeros(m * n * 2)
    for j in range(m):
        for i in range(n):
            Vot = dx[j, i] * dy[j, i] * dz[j, i] / props['time'].tstep * por / 5.615
            D[(mat[j, i] - 1) * 2] = Vot * (
                        (1 - sg_n1[j, i]) * b_o[j, i] - (1 - sg_n[j, i]) * props['fluid'].calc_bo(p_grids_n[j, i]))
            g1 = sg_n1[j, i] * b_g[j, i] - sg_n[j, i] * props['fluid'].calc_bg(p_grids_n[j, i])
            g2 = (1 - sg_n1[j, i]) * rs[j, i] * b_o[j, i] - (1 - sg_n[j, i]) * props['fluid'].calc_bo(p_grids_n[j, i]) * \
                 props['fluid'].calc_rs(p_grids_n[j, i])
            D[(mat[j, i] - 1) * 2 + 1] = Vot * (g1 + g2)
    return D


def distribute_properties(props):
    """
    Distribute rock and fluid properties to the grid
    :param props: dictionary of rock and fluid properties
    :return: dictionary of grids containing rock and fluid properties (after property distribution)
    """

    grid = props['grid']
    fluid = props['fluid']
    rock = props['rock']
    res = props['res']

    b_o = np.zeros(grid.Ny * grid.Nx)
    db_o = np.zeros(grid.Ny * grid.Nx)
    b_g = np.zeros(grid.Ny * grid.Nx)
    db_g = np.zeros(grid.Ny * grid.Nx)
    mu_g = np.zeros(grid.Ny * grid.Nx)
    dmu_g = np.zeros(grid.Ny * grid.Nx)
    rs = np.zeros(grid.Ny * grid.Nx)
    drs = np.zeros(grid.Ny * grid.Nx)
    kr_o = np.zeros(grid.Ny * grid.Nx)
    dkr_o = np.zeros(grid.Ny * grid.Nx)
    kr_g = np.zeros(grid.Ny * grid.Nx)
    dkr_g = np.zeros(grid.Ny * grid.Nx)
    for i in range(grid.Nx * grid.Ny):
        # Fluid and rock properties
        b_o[i] = fluid.calc_bo(res.press_n1_k[i])
        db_o[i] = fluid.calc_dbo(res.press_n1_k[i])
        b_g[i] = fluid.calc_bg(res.press_n1_k[i])
        db_g[i] = fluid.calc_dbg(res.press_n1_k[i])
        mu_o = np.full((grid.Ny, grid.Nx), fluid.mu_o)
        mu_g[i] = fluid.calc_mu_g(res.press_n1_k[i])
        dmu_g[i] = fluid.calc_dmu_g(res.press_n1_k[i])
        rs[i] = fluid.calc_rs(res.press_n1_k[i])
        drs[i] = fluid.calc_drs(res.press_n1_k[i])
        kr_o[i] = rock.calc_kro(res.sg_n1_k[i])
        dkr_o[i] = rock.calc_dkro(res.sg_n1_k[i])
        kr_g[i] = rock.calc_krg(res.sg_n1_k[i])
        dkr_g[i] = rock.calc_dkrg(res.sg_n1_k[i])

        # Grid size
        dx = np.full((grid.Ny, grid.Nx), grid.grid_dimension_x(res.Lx))
        dy = np.full((grid.Ny, grid.Nx), grid.grid_dimension_y(res.Ly))
        dz = np.full((grid.Ny, grid.Nx), grid.grid_dimension_z(res.Lz))
    params = {'k_x': rock.kx, 'k_y': rock.ky, 'b_o': b_o, 'db_o': db_o, 'b_g': b_g, 'db_g': db_g,
              'mu_o': mu_o, 'mu_g': mu_g, 'dmu_g': dmu_g, 'rs': rs, 'drs': drs, 'p_grids_n': res.press_n,
              'p_grids_n1': res.press_n1_k,
              'sg_n': res.sg_n, 'sg_n1': res.sg_n1_k, 'kr_o': kr_o, 'dkr_o': dkr_o, 'kr_g': kr_g, 'dkr_g': dkr_g,
              'dx': dx, 'dy': dy, 'dz': dz}
    for p in params:
        params[p] = np.reshape(params[p], (grid.Ny, grid.Nx))
    return params


def update_timestep(props, delta, eta_s, eta_p, omega, dt_max):
    dt_p = np.min((1 + omega) * eta_p / (delta[::2] + omega * eta_p))
    dt_s = np.min((1 + omega) * eta_s / (delta[1::2] + omega * eta_s))
    dt_upd = props['time'].tstep * np.min([dt_p, dt_s])
    return np.max([dt_upd, dt_max])


def update_parameters(mat, params, props):
    fluid = props['fluid']
    rock = props['rock']

    m = mat.shape[0]
    n = mat.shape[1]
    for j in range(m):
        for i in range(n):
            # Fluid and rock properties
            params['b_o'][j, i] = fluid.calc_bo(params['p_grids_n1'][j, i])
            params['db_o'][j, i] = fluid.calc_dbo(params['p_grids_n1'][j, i])
            params['b_g'][j, i] = fluid.calc_bg(params['p_grids_n1'][j, i])
            params['db_g'][j, i] = fluid.calc_dbg(params['p_grids_n1'][j, i])
            params['mu_g'][j, i] = fluid.calc_mu_g(params['p_grids_n1'][j, i])
            params['dmu_g'][j, i] = fluid.calc_dmu_g(params['p_grids_n1'][j, i])
            params['rs'][j, i] = fluid.calc_rs(params['p_grids_n1'][j, i])
            params['drs'][j, i] = fluid.calc_drs(params['p_grids_n1'][j, i])
            params['kr_o'][j, i] = rock.calc_kro(params['sg_n1'][j, i])
            params['dkr_o'][j, i] = rock.calc_dkro(params['sg_n1'][j, i])
            params['kr_g'][j, i] = rock.calc_krg(params['sg_n1'][j, i])
            params['dkr_g'][j, i] = rock.calc_dkrg(params['sg_n1'][j, i])
    return params


def construct_well_jacobian(mat, props, params):
    wells = props['well']
    grid = props['grid']

    k_x = params['k_x']
    k_y = params['k_y']
    kr_o = params['kr_o']
    dkr_o = params['dkr_o']
    kr_g = params['kr_g']
    dkr_g = params['dkr_g']
    b_o = params['b_o']
    db_o = params['db_o']
    b_g = params['b_g']
    db_g = params['db_g']
    mu_o = params['mu_o']
    mu_g = params['mu_g']
    dmu_g = params['dmu_g']
    rs = params['rs']
    drs = params['drs']

    dx = params['dx']
    dy = params['dy']
    dz = params['dz']
    p_grids_n1 = params['p_grids_n1']

    m = mat.shape[0]
    n = mat.shape[1]
    J_w = np.zeros((m * n * 2, m * n * 2))
    for well in wells:
        xc = well.loc[0] - 1
        yc = well.loc[1] - 1

        if well.qo_control == True:
            # Assign well flow elements to Jacobian matrix (oil rate control)
            J_w[(well.index_to_grid(grid.Nx) - 1) * 2 + 1, (well.index_to_grid(grid.Nx) - 1) * 2] = well.q_lim * (
                        kr_g[yc, xc] * mu_o[yc, xc] * db_g[yc, xc] / kr_o[yc, xc] / b_o[yc, xc] / mu_g[yc, xc] + kr_g[
                    yc, xc] * mu_o[yc, xc] * b_g[yc, xc] * db_o[yc, xc] / kr_o[yc, xc] / b_o[yc, xc] ** 2 / mu_g[
                            yc, xc] - kr_g[yc, xc] * mu_o[yc, xc] * b_g[yc, xc] * dmu_g[yc, xc] / kr_o[yc, xc] / b_o[
                            yc, xc] / mu_g[yc, xc] ** 2 + drs[yc, xc])
            J_w[(well.index_to_grid(grid.Nx) - 1) * 2 + 1, (well.index_to_grid(grid.Nx) - 1) * 2 + 1] = well.q_lim * \
                                                                                                        b_g[yc, xc] * \
                                                                                                        mu_o[yc, xc] / \
                                                                                                        b_o[yc, xc] / \
                                                                                                        mu_g[yc, xc] * (
                                                                                                                    dkr_g[
                                                                                                                        yc, xc] *
                                                                                                                    kr_o[
                                                                                                                        yc, xc] -
                                                                                                                    dkr_o[
                                                                                                                        yc, xc] *
                                                                                                                    kr_g[
                                                                                                                        yc, xc]) / \
                                                                                                        kr_o[
                                                                                                            yc, xc] ** 2
        else:
            # Assign well flow elements to Jacobian matrix (bottom hole pressure control)
            ro = 0.28 * ((k_y[yc, xc] / k_x[yc, xc]) ** 0.5 * dx[yc, xc] ** 2 + (k_x[yc, xc] / k_y[yc, xc]) ** 0.5 * dy[
                yc, xc] ** 2) ** 0.5 / ((k_y[yc, xc] / k_x[yc, xc]) ** 0.25 + (k_x[yc, xc] / k_y[yc, xc]) ** 0.25)
            WI = 2 * np.pi * (k_x[yc, xc] * k_y[yc, xc]) ** 0.5 * dz[yc, xc] / (np.log(ro / well.rw))

            J_w[(well.index_to_grid(grid.Nx) - 1) * 2, (well.index_to_grid(grid.Nx) - 1) * 2] = WI * 0.001127 * (
                        kr_o[yc, xc] * b_o[yc, xc] / mu_o[yc, xc] + kr_o[yc, xc] / mu_o[yc, xc] * db_o[yc, xc] * (
                            p_grids_n1[yc, xc] - well.pwf_lim))
            J_w[(well.index_to_grid(grid.Nx) - 1) * 2, (well.index_to_grid(grid.Nx) - 1) * 2 + 1] = WI * 0.001127 * (
                        b_o[yc, xc] / mu_o[yc, xc] * dkr_o[yc, xc] * (p_grids_n1[yc, xc] - well.pwf_lim))
            J_w[(well.index_to_grid(grid.Nx) - 1) * 2 + 1, (well.index_to_grid(grid.Nx) - 1) * 2] = WI * 0.001127 * (
                        kr_g[yc, xc] * b_g[yc, xc] / mu_g[yc, xc] + kr_g[yc, xc] * (
                            db_g[yc, xc] * mu_g[yc, xc] - b_g[yc, xc] * dmu_g[yc, xc]) / mu_g[yc, xc] ** 2 * (
                                    p_grids_n1[yc, xc] - well.pwf_lim) + b_o[yc, xc] * kr_o[yc, xc] * rs[yc, xc] / mu_o[
                            yc, xc] + kr_o[yc, xc] / mu_o[yc, xc] * (
                                    drs[yc, xc] * b_o[yc, xc] + rs[yc, xc] * db_o[yc, xc]) * (
                                    p_grids_n1[yc, xc] - well.pwf_lim))
            J_w[(well.index_to_grid(grid.Nx) - 1) * 2 + 1, (
                        well.index_to_grid(grid.Nx) - 1) * 2 + 1] = WI * 0.001127 * (
                        p_grids_n1[yc, xc] - well.pwf_lim) * (b_g[yc, xc] / mu_g[yc, xc] * dkr_g[yc, xc] + rs[yc, xc] *
                                                              b_o[yc, xc] / mu_o[yc, xc] * dkr_o[yc, xc])
    return J_w


def construct_well_residual(mat, props, params):
    wells = props['well']
    grid = props['grid']

    k_x = params['k_x']
    k_y = params['k_y']
    kr_o = params['kr_o']
    kr_g = params['kr_g']
    b_o = params['b_o']
    b_g = params['b_g']
    mu_o = params['mu_o']
    mu_g = params['mu_g']
    rs = params['rs']

    dx = params['dx']
    dy = params['dy']
    dz = params['dz']
    p_grids_n1 = params['p_grids_n1']

    m = mat.shape[0]
    n = mat.shape[1]
    Q = np.zeros((m * n * 2,))

    for well in wells:
        xc = well.loc[0] - 1
        yc = well.loc[1] - 1

        if well.qo_control == True:
            # Assign well flow elements to Jacobian matrix (oil rate control)
            Q[(well.index_to_grid(grid.Nx) - 1) * 2] = well.q_lim
            Q[(well.index_to_grid(grid.Nx) - 1) * 2 + 1] = well.q_lim * (
                        kr_g[yc, xc] / kr_o[yc, xc] * b_g[yc, xc] / b_o[yc, xc] * mu_o[yc, xc] / mu_g[yc, xc] + rs[
                    yc, xc])
        else:
            # Assign well flow elements to Jacobian matrix (bottom hole pressure control)
            ro = 0.28 * ((k_y[yc, xc] / k_x[yc, xc]) ** 0.5 * dx[yc, xc] ** 2 + (k_x[yc, xc] / k_y[yc, xc]) ** 0.5 * dy[
                yc, xc] ** 2) ** 0.5 / ((k_y[yc, xc] / k_x[yc, xc]) ** 0.25 + (k_x[yc, xc] / k_y[yc, xc]) ** 0.25)
            WI = 2 * np.pi * (k_x[yc, xc] * k_y[yc, xc]) ** 0.5 * dz[yc, xc] / (np.log(ro / well.rw))

            Q[(well.index_to_grid(grid.Nx) - 1) * 2] = WI * 0.001127 * b_o[yc, xc] * kr_o[yc, xc] / mu_o[yc, xc] * (
                        p_grids_n1[yc, xc] - well.pwf_lim)
            Q[(well.index_to_grid(grid.Nx) - 1) * 2 + 1] = WI * 0.001127 * (
                        kr_g[yc, xc] * b_g[yc, xc] / mu_g[yc, xc] * (p_grids_n1[yc, xc] - well.pwf) + rs[yc, xc] * b_o[
                    yc, xc] * kr_o[yc, xc] / mu_o[yc, xc] * (p_grids_n1[yc, xc] - well.pwf_lim))
    return Q


def calc_rate(props, params, well_no):
    k_x = params['k_x']
    k_y = params['k_y']

    dx = params['dx']
    dy = params['dy']
    dz = params['dz']

    xc = props['well'][well_no].loc[0] - 1
    yc = props['well'][well_no].loc[1] - 1
    ro = 0.28 * ((k_y[yc, xc] / k_x[yc, xc]) ** 0.5 * dx[yc, xc] ** 2 + (k_x[yc, xc] / k_y[yc, xc]) ** 0.5 * dy[
        yc, xc] ** 2) ** 0.5 / ((k_y[yc, xc] / k_x[yc, xc]) ** 0.25 + (k_x[yc, xc] / k_y[yc, xc]) ** 0.25)
    WI = 2 * np.pi * (k_x[yc, xc] * k_y[yc, xc]) ** 0.5 * dz[yc, xc] / np.log(ro / props['well'][well_no].rw) * 0.001127
    props['well'][well_no].q = WI * params['kr_o'][yc, xc] * params['b_o'][yc, xc] / params['mu_o'][yc, xc] * (
                params['p_grids_n1'][yc, xc] - props['well'][well_no].pwf)
    return props['well'][well_no].q


def calc_gas_rate(props, params, well_no):
    k_x = params['k_x']
    k_y = params['k_y']

    dx = params['dx']
    dy = params['dy']
    dz = params['dz']

    xc = props['well'][well_no].loc[0] - 1
    yc = props['well'][well_no].loc[1] - 1
    ro = 0.28 * ((k_y[yc, xc] / k_x[yc, xc]) ** 0.5 * dx[yc, xc] ** 2 + (k_x[yc, xc] / k_y[yc, xc]) ** 0.5 * dy[
        yc, xc] ** 2) ** 0.5 / ((k_y[yc, xc] / k_x[yc, xc]) ** 0.25 + (k_x[yc, xc] / k_y[yc, xc]) ** 0.25)
    WI = 2 * np.pi * (k_x[yc, xc] * k_y[yc, xc]) ** 0.5 * dz[yc, xc] / np.log(ro / props['well'][well_no].rw) * 0.001127
    qg = WI * (params['kr_o'][yc, xc] * params['b_o'][yc, xc] / params['mu_o'][yc, xc] * params['rs'][yc, xc] +
               params['kr_g'][yc, xc] * params['b_g'][yc, xc] / params['mu_g'][yc, xc]) * (
                     params['p_grids_n1'][yc, xc] - props['well'][well_no].pwf) / (1000 / 5.615)
    return qg


def calc_pwf(props, params, well_no):
    k_x = params['k_x']
    k_y = params['k_y']

    dx = params['dx']
    dy = params['dy']
    dz = params['dz']

    xc = props['well'][well_no].loc[0] - 1
    yc = props['well'][well_no].loc[1] - 1
    ro = 0.28 * ((k_y[yc, xc] / k_x[yc, xc]) ** 0.5 * dx[yc, xc] ** 2 + (k_x[yc, xc] / k_y[yc, xc]) ** 0.5 * dy[
        yc, xc] ** 2) ** 0.5 / ((k_y[yc, xc] / k_x[yc, xc]) ** 0.25 + (k_x[yc, xc] / k_y[yc, xc]) ** 0.25)
    WI = 2 * np.pi * (k_x[yc, xc] * k_y[yc, xc]) ** 0.5 * dz[yc, xc] / np.log(ro / props['well'][well_no].rw) * 0.001127
    props['well'][well_no].pwf = params['p_grids_n1'][yc, xc] - props['well'][well_no].q / WI / params['kr_o'][yc, xc] / \
                                 params['b_o'][yc, xc] * params['mu_o'][yc, xc]
    return props['well'][well_no].pwf


def calc_cfl(qo, dt, params, props):
    PV = params['dx'][qo.loc[0] - 1, qo.loc[1] - 1] * params['dy'][qo.loc[0] - 1, qo.loc[1] - 1] * params['dz'][
        qo.loc[0] - 1, qo.loc[1] - 1] * props['rock'].por
    # qt = qo.q + qo.q*params['kr_g'][qo.loc[0]-1, qo.loc[1]-1]/params['kr_o'][qo.loc[0]-1, qo.loc[1]-1]*params['b_g'][qo.loc[0]-1, qo.loc[1]-1]/params['b_o'][qo.loc[0]-1, qo.loc[1]-1]*params['mu_o'][qo.loc[0]-1, qo.loc[1]-1]/params['mu_g'][qo.loc[0]-1, qo.loc[1]-1]
    qt = qo.q + qo.q * params['rs'][qo.loc[0] - 1, qo.loc[1] - 1]
    cfl = qt * props['time'].tstep / PV
    return cfl


def run_simulation(mat, props, params):
    grid = props['grid']
    res = props['res']
    wells = props['well']

    # Set tolerances for Newton's convergence
    eps_1 = 1e-3  # Infinite norm Residual tolerance
    eps_2 = 0.01  # Gas saturation tolerance
    eps_3 = 0.001  # Pressure tolerance

    # Set auto time stepping parameters
    eta_s = 0.05
    eta_p = 50
    omega = 0.5
    dt_max = 4

    y_int = props['well'][0].loc[0] - 1
    x_int = props['well'][0].loc[1] - 1
    t_list = []
    p_well_block = []
    qo_list = []
    qg_list = []
    fpr_list = []
    bhp_list = []
    nr_list = []
    cfl_list = []
    t = 0
    p_n = res.stack_ps(res.press_n, res.sg_n)

    while t < props['time'].tmax:
        # Store variables of interest
        p_well_block.append(params['p_grids_n1'][y_int, x_int])
        fpr_list.append(np.average(params['p_grids_n1']))
        bhp_list.append(wells[0].pwf)
        t_list.append(t)
        qo_list.append(calc_rate(props, params, well_no=0))
        qg_list.append(calc_gas_rate(props, params, well_no=0))

        # Newton's Iteration Loop
        crit1 = 99
        crit2 = 99
        crit3 = 99

        # Parameters preconditioning
        p_n1_k = p_n
        params['p_grids_n1'] = params['p_grids_n']
        params['sg_n1'] = params['sg_n']
        props['res'].press_n1_k = props['res'].press_n
        props['res'].sg_n1_k = props['res'].sg_n

        # Newton's Iteration Loop
        max_nr_iter = 15
        i = 0
        while crit1 > eps_1 or (crit2 >= eps_2 and crit3 >= eps_3):
            # while crit1 > eps_1:
            if i <= max_nr_iter:
                # Update flow rate and pressure
                for j, well in enumerate(wells):
                    if well.pwf < well.pwf_lim:
                        well.qo_control = False
                    if well.qo_control == False:
                        well.pwf = well.pwf_lim
                        well.q = calc_rate(props, params, well_no=j)
                    else:
                        well.q = well.q_lim
                        well.pwf = calc_pwf(props, params, well_no=j)

                # Construct matrix T (containing transmissibility terms)
                T = construct_T(mat, params)

                # Construct matrix D (containing accumulation terms)
                D = construct_D(mat, params, props)

                # Compute residual matrix
                Q = construct_well_residual(mat, props, params)
                R = np.dot(T, p_n1_k) - D - Q

                # Compute Jacobian Matrix
                J_ori = construct_J(mat, params, props, T)
                J_w = construct_well_jacobian(mat, props, params)
                J = J_ori - J_w

                # Update the solution
                delta = (gmres(J, -R))[0]
                p_n1_k1 = p_n1_k + delta
                props['res'].press_n1_k = p_n1_k1[::2]
                props['res'].sg_n1_k = p_n1_k1[1::2]
                params['p_grids_n1'] = np.reshape(props['res'].press_n1_k, (grid.Ny, grid.Nx))
                params['sg_n1'] = np.reshape(props['res'].sg_n1_k, (grid.Ny, grid.Nx))

                # Update rock and fluid properties
                params = update_parameters(mat, params, props)

                # Update flow rate and bottom hole pressure
                for j, well in enumerate(wells):
                    if well.qo_control == False:
                        well.pwf = well.pwf_lim
                        well.q = calc_rate(props, params, well_no=j)
                    else:
                        well.q = well.q_lim
                        well.pwf = calc_pwf(props, params, well_no=j)

                # Assess convergence criteria
                PV = params['dx'] * params['dy'] * params['dz'] * props['rock'].por
                crit1_o = np.abs(
                    5.615 / np.reshape(params['b_o'], (grid.Nx * grid.Ny,)) * props['time'].tstep * R[::2] / np.reshape(
                        PV, (grid.Nx * grid.Ny,)))
                crit1_g = np.abs(5.615 / np.reshape(params['b_g'], (grid.Nx * grid.Ny,)) * props['time'].tstep * R[
                                                                                                                 1::2] / np.reshape(
                    PV, (grid.Nx * grid.Ny,)))
                crit1 = np.max([crit1_o, crit1_g])
                crit2 = np.max(np.abs(p_n1_k1[1::2] - p_n1_k[1::2]))
                crit3 = np.max(np.abs((p_n1_k1[::2] - p_n1_k[::2]) / np.average(p_n1_k1[::2])))

                # Update time-step
                # props['time'].tstep = update_timestep(props, delta, eta_s, eta_p, omega, dt_max)

                p_n1_k = p_n1_k1
                print('timestep : %.3f; NR iter : %.0f; Residual Error: %.4f' % (t, i + 1, crit1))
                i += 1
            else:
                print('entering else')
                p_n1_k = p_n
                props['time'].tstep /= 2
                i = 0
        p_n = p_n1_k1
        params['p_grids_n'] = params['p_grids_n1']
        params['sg_n'] = params['sg_n1']
        props['res'].press_n = props['res'].press_n1_k
        props['res'].sg_n = props['res'].sg_n1_k
        t += props['time'].tstep
        nr_list.append(i)
        cfl_list.append(calc_cfl(wells[0], props['time'].tstep, params, props))

    return [t_list, qo_list, qg_list, bhp_list, p_well_block, fpr_list, nr_list, cfl_list]


def plot_pressure(t, p_pred, label, color):
    """Plots pressure v time"""
    plt.plot(t, p_pred, color, label=label, linewidth=3)
    plt.xlabel("Time (days)")
    plt.ylabel("Pressure (psi)", fontsize=9)
    plt.legend(loc="best", prop=dict(size=8))
    plt.xlim(0, max(t))
    plt.ylim(min(p_pred), max(p_pred))
    plt.grid(True)
    plt.draw()


def plot_rate(t, q_pred, label, color):
    """Plots rate v time"""
    plt.plot(t, q_pred, color, label=label, linewidth=3)
    plt.xlabel("Time (days)")
    plt.ylabel("Oil Rate (STB/day)", fontsize=9)
    plt.legend(loc="best", prop=dict(size=8))
    plt.xlim(0, max(t))
    plt.ylim(0, max(q_pred))
    plt.grid(True)
    plt.draw()


def plot_gas_rate(t, q_pred, label, color):
    """Plots rate v time"""
    plt.plot(t, q_pred, color, label=label, linewidth=3)
    plt.xlabel("Time (days)")
    plt.ylabel("Gas Rate (MSCFD)", fontsize=9)
    plt.legend(loc="best", prop=dict(size=8))
    plt.xlim(0, max(t))
    plt.ylim(0, max(q_pred))
    plt.grid(True)
    plt.draw()


def plot_var(t, var_pred, y_axis_title, label, color):
    """Plots rate v time"""
    plt.plot(t, var_pred, color, label=label, linewidth=3)
    plt.xlabel("Time (days)")
    plt.ylabel(y_axis_title, fontsize=9)
    plt.legend(loc="best", prop=dict(size=8))
    plt.xlim(0, max(t))
    plt.ylim(0, max(var_pred))
    plt.grid(True)
    plt.draw()


def spatial_map(p_2D, title):
    """Plots variable of interest on a 2D spatial map"""
    plt.matshow(p_2D)
    plt.colorbar()
    plt.xlabel('grid in x-direction')
    plt.ylabel('grid in y-direction')
    plt.title(title)
    plt.draw()
