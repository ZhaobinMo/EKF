import numpy as np
from filterpy.kalman import ExtendedKalmanFilter as EKF
from numpy import dot, zeros, eye

# g function in the flux-function
def g(x, para):
    lamda = para['lambda']
    p = para['p']
    temp = 1 + lamda ** 2 * (x - p) ** 2
    return np.sqrt(temp)


def dg(x, para):
    lamda = para['lambda']
    p = para['p']
    numer = lamda ** 2 * (x - p)
    denum = g(x, para)
    return numer / denum

# flux function
def q_of(rho, para):
    rho_max = para['rho_max']
    alpha = para['alpha']
    #
    a = g(0, para)
    b = g(1, para)
    c = g(rho / rho_max, para)
    Q = alpha * (a + (b - a) * rho / rho_max - c)

    # a = np.sqrt(1 + (lamda * p) ** 2)
    # b = np.sqrt(1 + (lamda * (1 - p)) ** 2)
    # y = lamda * (rho / rho_max - p)
    # Q = alpha * (a + (b - a) * rho / rho_max - np.sqrt(1 + y ** 2))
    return Q


def dq_of(rho, para):
    rho_max = para['rho_max']
    alpha = para['alpha']

    a = (g(1, para) - g(0, para)) / rho_max
    b = dg(rho / rho_max, para) / rho_max

    return alpha * (a - b)


# H

def h_at(x, loops):
    result = np.array([x[i] for i in loops])
    return np.array([x[i] for i in loops])


# lf

def f_at(x, **kwargs):
    dx = kwargs['dx']
    dt = kwargs['dt']
    para = kwargs['para']
    x_l = np.zeros(x.shape)
    x_l[0] = x[0]
    x_l[1:] = x[:-1]

    x_r = np.zeros(x.shape)
    x_r[-1] = x[-1]
    x_r[:-1] = x[1:]

    # print('x:',x)
    # print('x_l', x_l)
    # print('x_r', x_r)

    #q_l = np.array(list(map(q_of, x_l)))
    q_l = np.array([q_of(x, para) for x in x_l])
    #q_r = np.array(list(map(q_of, x_r)))
    q_r = np.array([q_of(x, para) for x in x_r])

    q_pred = (x_l + x_r) / 2 - (dt / 2 / dx) * (q_r - q_l)
    return q_pred


def Hjacobian(x, N, loops):
    H_jacobian = np.zeros( (len(loops),N) )
    for i in range(len(loops)):
        H_jacobian[i, loops[i]] = 1
    return H_jacobian

def partial_r(x_i, **kwargs):
    dx = kwargs['dx']
    dt = kwargs['dt']
    para = kwargs['para']
    return 1/2 + dt/dx/2*dq_of(x_i, para)
def partial_l(x_i, **kwargs):
    dx = kwargs['dx']
    dt = kwargs['dt']
    para = kwargs['para']
    return 1/2 - dt/dx/2*dq_of(x_i, para)

def Fjacobian(x, **f_kwargs):
    F = np.zeros((len(x), len(x)))
    F[0, 0] = partial_l(x[0],**f_kwargs)
    F[0, 1] = partial_r(x[1],**f_kwargs)

    F[-1, -1] = partial_r(x[-1], **f_kwargs)
    F[-1, -2] = partial_l(x[-2], **f_kwargs)
    for i in range(1, len(x) - 1):
        F[i, i - 1] = partial_l(x[i - 1], **f_kwargs)
        F[i, i + 1] = partial_r(x[i + 1], **f_kwargs)
    return F

def neibour_avg(x):
    print('neibour is used')
    for i in range(1, len(x)-1):
        x[i] = np.mean([x[i-1], x[i], x[i+1]])
    return x


class LWREKF(EKF):
    def __init__(self, dim_x, dim_z, std_pred, std_update):
        EKF.__init__(self, dim_x, dim_z)
        self.std_pred = std_pred
        self.std_update = std_update
        self.Q = eye(dim_x) * std_pred
        self.R = eye(dim_z) * std_update

    def predict_x(self, f_at, **f_kwargs):
        self.x = f_at(self.x, **f_kwargs)

    def predict(self, FJacobian, f_at, **f_kwargs):
        self.F = FJacobian(self.x, **f_kwargs)
        self.predict_x(f_at, **f_kwargs)

        self.P = dot(self.F, self.P).dot(self.F.T) + self.Q

        # save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)

def implement_kf(N, loops, rho, config, **f_kwargs):
    std_Q = config['std_Q']
    std_R = config['std_R']
    init_rho = config['init_rho']
    init_P = config['init_P']

    lwrefk = LWREKF(N,len(loops),std_Q, std_R)
    lwrefk.x = init_rho
    lwrefk.P = init_P
    X_pri = [lwrefk.x]
    X_pos = [lwrefk.x]
    Obs = []
    K = []
    P = []
    for i in range(rho.shape[1] - 1):
        lwrefk.predict(Fjacobian, f_at, **f_kwargs)
        X_pri.append(lwrefk.x_prior)

        if i % 1 == 0:
            observe = [rho[k, i + 1] for k in loops]
            Obs.append(observe)
            lwrefk.update(observe, Hjacobian, h_at, args=(N, loops),hx_args=(loops))
            #lwrefk.x = neibour_avg(lwrefk.x)
            lwrefk.x_post = lwrefk.x

        else:
            # X_pos.append(lwrefk.x_prior)
            pass
        K.append(lwrefk.K)
        P.append(lwrefk.P)
        X_pos.append(lwrefk.x_post)
    X_pos = np.vstack(X_pos).T
    X_pri = np.vstack(X_pri).T

    return X_pri, X_pos, K, P
