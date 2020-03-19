import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from math import log, exp, sqrt
import sys
import numpy as np
from numpy import dot, zeros, eye
import scipy.linalg as linalg
from filterpy.stats import logpdf
from filterpy.common import pretty_str, reshape_z
import seaborn as sns

rho = np.loadtxt('./data/lf/rho_bell_mag_075.csv',delimiter=',')
rho = rho[:,:-1]

u = np.loadtxt('./data/lf/u_bell_mag_075.csv',delimiter=',')
print(rho.shape, u.shape)

#rho = rho[::10, ::10]
u = u[::10, ::10]
N, T = rho.shape

# LF solver
U_MAX = 1
RHO_MAX = 1

dt = 3/T
dx = 1/N

# loop index
loops = [i for i in range(N) if i%30 == 0]
print('loops: ', loops)

def u_of(x):
    return U_MAX*(1-x/RHO_MAX)

def q_of(x):
    return x*u_of(x)


# H

def h_at(x):
    return np.array([x[i] for i in loops])


# lf

def f_at(x):
    x_l = np.zeros(x.shape)
    x_l[0] = x[-1]
    x_l[1:] = x[:-1]

    x_r = np.zeros(x.shape)
    x_r[-1] = x[0]
    x_r[:-1] = x[1:]

    # print('x:',x)
    # print('x_l', x_l)
    # print('x_r', x_r)

    q_l = np.array(list(map(q_of, x_l)))
    q_r = np.array(list(map(q_of, x_r)))

    q_pred = (x_l + x_r) / 2 - (dt / 2 / dx) * (q_r - q_l)
    return q_pred

def Hjacobian(x, *args):
    H_jacobian = np.zeros( (len(loops), N) )
    for i in range(len(loops)):
        H_jacobian[i, loops[i]] = 1
    return H_jacobian


def partial_r(x_i):
    return 1 / 2 + dt / dx / 2 * U_MAX * (2 * x_i / RHO_MAX - 1)


def partial_l(x_i):
    return 1 / 2 + dt / dx / 2 * U_MAX * (1 - 2 * x_i / RHO_MAX)


def Fjacobian(x):
    F = np.zeros((len(x), len(x)))
    F[0, 0] = partial_l(x[0])
    F[0, 1] = partial_r(x[1])

    F[-1, -1] = partial_r(x[-1])
    F[-1, -2] = partial_r(x[-2])
    for i in range(1, len(x) - 1):
        F[i, i - 1] = partial_l(x[i - 1])
        F[i, i + 1] = partial_r(x[i + 1])
    return F


from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import ExtendedKalmanFilter as EKF
#rk = ExtendedKalmanFilter(dim_x=N, dim_z=N)

class LWREKF(EKF):
    def __init__(self, dim_x, dim_z, std_pred, std_update):
        EKF.__init__(self, dim_x, dim_z)
        self.std_pred = std_pred
        self.std_update = std_update
        self.Q = eye(dim_x) * std_pred
        self.R = eye(dim_z) * std_update

    def predict_x(self, f_at):
        self.x = f_at(self.x)

    def predict(self, FJacobian, f_at):
        self.F = FJacobian(self.x)
        self.predict_x(f_at)

        self.P = dot(self.F, self.P).dot(self.F.T) + self.Q

        # save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)


lwrefk = LWREKF(N, len(loops), 0.01, 0.01)
lwrefk.x = rho[:, 0]
X_pri = []
X_pos = []
K = []
for i in range(rho.shape[1] - 1):
    lwrefk.predict(Fjacobian, f_at)
    X_pri.append(lwrefk.x_prior)

    if i%10 == 0:
        observe = [rho[k, i + 1] for k in loops]
        lwrefk.update(observe, Hjacobian, h_at)
        X_pos.append(lwrefk.x_post)
    else:
        pass
    K.append(lwrefk.K)

X_pos = np.vstack(X_pos).T
sns.heatmap(X_pos, vmin = 0, vmax=1)
plt.savefig('./data/out/heatmap/pos.png')