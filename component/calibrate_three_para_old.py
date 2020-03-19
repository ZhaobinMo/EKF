import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.dates as mdates
from scipy import optimize

import pickle
with open('./data/NGSIM/US101_lane1_t30s30.pickle', "rb") as f:
    rho_q = pickle.load(f)
    rho_mat = rho_q['rhoMat']
    q_mat = rho_q['qMat']
class Looper():
    def __init__(self, rho_mat, q_mat):
        assert rho_mat.shape == q_mat.shape
        self.rho_mat = rho_mat
        self.q_mat = q_mat
        self.L = rho_mat.shape[0]
        self.T = rho_mat.shape[1]
        self.init_pos()
        print("L=", self.L)
        print("T=", self.T)

    def init_pos(self, offset=0, n_loop=3):
        self.off_set = offset
        self.n_loop = n_loop

    def _get_loop_data(self):
        st = 0  # index
        ed = self.L   # index

        loop_locs = [round((self.off_set + i / self.n_loop) * ed) for i in range(self.n_loop)]
        loop_locs = list(map(lambda x: x - ed if x >= ed else x, loop_locs))
        print('location of loop:', loop_locs)
        self.rho_loops = [rho_mat[i, :] for i in loop_locs]
        self.q_loops = [q_mat[i, :] for i in loop_locs]
        self.loop_locs = loop_locs

    def _Q(self, rho, lamda, p, rho_max, alpha):
        a = np.sqrt(1 + (lamda * p) ** 2)
        b = np.sqrt(1 + (lamda * (1 - p)) ** 2)
        y = lamda * (rho / rho_max - p)
        Q = alpha * (a + (b - a) * rho / rho_max - np.sqrt(1 + y ** 2))
        return Q

    def _calibrate(self, rho, q):
        assert len(rho) == len(q)
        rho = np.array(rho)
        q = np.array(q)
        x = rho[~(np.isnan(rho) | np.isnan(q))]
        y = q[~(np.isnan(rho) | np.isnan(q))]
        popt, pcov = optimize.curve_fit(self._Q, x, y, maxfev=10000)
        return popt

    def calibrate(self, verbose=True):
        self._get_loop_data()

        popts = list(map(self._calibrate, self.rho_loops, self.q_loops))

        # all data
        rho_all = [item for sublist in self.rho_loops for item in sublist]
        q_all = [item for sublist in self.q_loops for item in sublist]
        popt_all = self._calibrate(rho_all, q_all)

        if verbose:
            for idx, popt in enumerate(popts):
                print("------ loop:{} at pos={} ------".format(idx, self.loop_locs[idx]))
                print('lambda = {} \np = {} \nrho_max = {} \nalpha = {}'.format(popt[0]
                                                                                , popt[1]
                                                                                , popt[2]
                                                                                , popt[3]))
            print("======= all loop together ========")
            print('lambda = {} \np = {} \nrho_max = {} \nalpha = {}'.format(popt_all[0]
                                                                            , popt_all[1]
                                                                            , popt_all[2]
                                                                            , popt_all[3]))
        return self.loop_locs, popts, popt_all


if __name__ == '__main__':
	
    with open("US101_lane1_t30s30_rvsPara.pickle", "rb") as f:
        rho_q = pickle.load(f)
    rho_mat = rho_q['rhoMat']
    q_mat = rho_q['qMat']





    looper = Looper(rho_mat, q_mat)

    #######################################################################################
    ### here change the number of loops, and the offset of the first loop (offset = [0,1])
    looper.init_pos(offset=0, n_loop=2)
    #######################################################################################
    loop_locs, popts, popt_all = looper.calibrate()

"""
when use all the data, the parameters are:
lambda = 13.353863348683868 
p = 0.18338128373498327 
rho_max = 0.14825264675390998 
alpha = 0.13500317129063663
"""
