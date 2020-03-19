import numpy as np
from filterpy.kalman import ExtendedKalmanFilter as EKF

import seaborn as sns
from copy import deepcopy
from math import log, exp, sqrt
import sys
import numpy as np
from numpy import dot, zeros, eye
import scipy.linalg as linalg
from filterpy.stats import logpdf
from filterpy.common import pretty_str, reshape_z
import os, sys
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import ExtendedKalmanFilter as EKF
import pickle
import matplotlib.pyplot as plt
from copy import deepcopy

from component.calibrate_three_para import Looper

plt.rcParams['font.size'] = 16
from component.ekf import *
import seaborn as sns



# NGSIM DATA
with open('./data/NGSIM/US101_lane1_t30s30.pickle', 'rb') as f:
    data = pickle.load(f)


rho_22 = data['rhoMat']
q_22 = data['qMat']

Aggregate_5 = False # True if aggregate the first 20 x_grid into 5
if Aggregate_5:
    rho = np.zeros((5, rho.shape[1]))
    q = np.zeros((5, rho.shape[1]))
    for i in range(5):
        for j in range(rho.shape[1]):
            sub_rho = rho_22[i:i+5,j]
            sub_q = q_22[i:i+5]
            rho[i,j] = np.average(sub_rho)
            q[i,j] = np.average(sub_q)
            dx = (data['s'][1] - data['s'][0])*5
else:
    rho = rho_22
    q = q_22
    
    dx = data['s'][1] - data['s'][0]

dt = data['t'][1] - data['t'][0]
#dt = dt / 6
#para = para_t30_s30

N = rho.shape[0]
T = rho.shape[1]
print('dt=', dt)
print('dx=', dx)
print('N=', N)
print('T=', T)
print('dx/dt', dx/dt)

# loops

LOOPS = {'2':[0, 21],
         '4':[0, 5, 11, 21],
         '6':[0, 3, 7, 11, 14, 21],
         '8':[0, 2, 5, 8, 11, 13, 16, 21],
         '10':[0, 2, 4, 6, 8, 11, 13, 15, 17, 21],
         '12':[0, 1, 3, 5, 7, 9, 11, 12, 14, 16, 18, 21],
         '14':[0, 1, 3, 4, 6, 7, 9, 11, 12, 14, 15, 17, 18, 21],
         '16':[0, 1, 2, 4, 5, 6, 8, 9, 11, 12, 13, 15, 16, 17, 19, 21],
         '18':[0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 17, 18, 19, 21]}

# implement
def implement_kf(N, loops, rho, config, **f_kwargs):
    std_Q = config['std_Q']
    std_R = config['std_R']
    init_rho = config['init_rho']
    init_P = config['init_P']

    lwrefk = LWREKF(N, len(loops), std_Q, std_R)
    lwrefk.x = init_rho
    lwrefk.P = init_P
    X_pri = [lwrefk.x]
    X_pos = [lwrefk.x]
    Obs = []
    K = []
    P = []
    for i in range((rho.shape[1] - 1) * 6):
        lwrefk.predict(Fjacobian, f_at, **f_kwargs)
        X_pri.append(lwrefk.x_prior)

        if i % 6 == 0:
            observe = [rho[k, int(i / 6) + 1] for k in loops]
            Obs.append(observe)

        lwrefk.update(observe, Hjacobian, h_at, args=(N, loops), hx_args=(loops))
        # lwrefk.x = neibour_avg(lwrefk.x)
        lwrefk.x_post = lwrefk.x
        if i % 6 == 0:
            X_pos.append(lwrefk.x_post)
        # else:
        # X_pos.append(lwrefk.x_prior)
        #    pass
        K.append(lwrefk.K)
        P.append(lwrefk.P)

    X_pos = np.vstack(X_pos).T
    X_pri = np.vstack(X_pri).T

    return X_pri, X_pos, K, P


ReadRealPara = True # True if the groung-truth LWR parameters are given

# KF
config_kf_init = {'std_Q':1,
                  'std_R':1,
                  #'init_rho': np.ones(rho.shape[0])*0.03,
                  'init_rho': np.array([0.03]*rho.shape[0]),
                  'init_P': np.eye(rho.shape[0])}

looper = Looper(rho, q)

Errors = []

#para
fitted_paras = []

for loops in LOOPS.values():
    
    #loops = [0,20]
    # para
    if ReadRealPara:
        para = data['para']
    else:
        #loop
        looper.init_pos(loops)
        loop_locs, popts, popt_all = looper.calibrate(verbose=False)
        para = {'lambda': popt_all[0],
               'p': popt_all[1],
               'rho_max': popt_all[2],
               'alpha': popt_all[3]}
        print('lambda = {} \np = {} \nrho_max = {} \nalpha = {}'.format(popt_all[0]
                                                                            , popt_all[1]
                                                                            , popt_all[2]
                                                                            , popt_all[3]))
        fitted_paras.append(popt_all)
    
    #loop
    X_pri, X_pos, K, P = implement_kf(N, loops, rho, config_kf_init,
                                dx=dx,
                                dt=dt,
                                para=para)
    error_rho = np.linalg.norm(rho[:, 10:]-X_pos[:, 10:],2)/np.linalg.norm(X_pos[:, 10:],2)
    Errors.append(error_rho)
    #break

plt.plot(Errors, '-*r')
ax = plt.gca()
ax.set_xticks(list(range(9)))
ax.set_xticklabels(list(LOOPS.keys()))
plt.title('fitted parameter')
plt.ylabel('error')
plt.xlabel('loops #')
plt.yticks([i for i in np.arange(0,1.1,0.1)])
plt.savefig('ground_truth_res.png')
