"""
@author: Rongye Shi
"""

import sys
sys.path.insert(0, 'Utilities/')

import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import scipy.io
from scipy.interpolate import griddata
from plotting import newfig, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import time
import pickle

if 1:
    np.random.seed(1234)
    tf.set_random_seed(1234)
else:
    np.random.seed(4321)
    tf.set_random_seed(4321)

#with open('US101_lane1_t10s30.pickle','rb') as f:
#with open('US101_lane1_t30s30.pickle','rb') as f:
with open('US101_lane1_t30s30.pickle','rb') as f:
    data_pickle = pickle.load(f)

lamb = data_pickle['para']['lambda']
p = data_pickle['para']['p']
rho_max = data_pickle['para']['rho_max']
alpha = data_pickle['para']['alpha']

a = np.sqrt( 1+(lamb*p)**2 )
b = np.sqrt( 1+(lamb*(1-p))**2 )

Loop_Num =10

t = sfdsf
class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, X, X2, rho, layers, lb, ub):
        
        self.lamb = lamb
        self.p = p
        self.rho_max = rho_max
        self.alpha = alpha
        self.rho_rev = 1.0/self.rho_max 

        self.A = a
        self.B = b
        
        self.lb = lb
        self.ub = ub
        
        self.x = X[:,0:1]
        self.x2 = X2[:,0:1]#colocation data point
        
        self.t = X[:,1:2]
        self.t2 = X2[:,1:2]
        self.rho = rho
        
        
        self.layers = layers
        
        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)
        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        
        
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]]) # self.x.shape[1] =1, None indicates that the first dimension, corresponding to the batch size, can be of any size
        self.x2_tf = tf.placeholder(tf.float32, shape=[None, self.x2.shape[1]])#colocation point x
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.rho_tf = tf.placeholder(tf.float32, shape=[None, self.rho.shape[1]])
        
        self.t2_tf = tf.placeholder(tf.float32, shape=[None, self.t2.shape[1]])#colocation point t
        
        
        
        
        self.rho_pred = self.net_rho(self.x_tf, self.t_tf)
        self.f_pred = self.net_f(self.x2_tf, self.t2_tf)
        
        '''
        self.loss = 100*tf.reduce_mean(tf.square(self.rho_tf - self.rho_pred)) + \
                    0.0*tf.reduce_mean(tf.square(self.f_pred))# + \
                    #1*tf.reduce_mean(tf.square(self.bound_pred))
        '''
        self.loss = 35*tf.reduce_mean(tf.square(self.rho_tf - self.rho_pred)) + \
                    0*tf.reduce_mean(tf.square(self.f_pred))
        
        self.f_loss = tf.reduce_mean(tf.square(self.f_pred))
        
        
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 5000000,#Maximum number of iterations
                                                                           'maxfun': 5000000, #Maximum number of function evaluations
                                                                           'maxcor': 50, # number of limited memory matric
                                                                           'maxls': 50, 
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
                                                    
                                                                           
        
        
        #self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate = 0.05)
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        
        
        init = tf.global_variables_initializer() # after this variables hold the values you told them to hold when you declare them will be made
        self.sess.run(init) # initialization run

    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1): # need the -1 because the first and last number in defining the nn is input and output
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases): # NP
        
        num_layers = len(weights) + 1 # need +1, because # of weights is one dim smaller
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0  # this is all element-wise operation, centralize and standardize the input data
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        #Y = tf.exp(tf.add(tf.matmul(H, W), b)) # the final layer has no activation func, so it has to be separetly processed
        Y = tf.add(tf.matmul(H, W), b)

        return Y
            
    def net_rho(self, x, t):  
        rho = self.neural_net(tf.concat([x,t],1), self.weights, self.biases) # concatenate is needed for making [[x0,t0],[x1,t1],...]
        return rho
    
    def Q_of_rho(self, rho):
        y = self.lamb * (self.rho_rev * rho - self.p)
        third = tf.sqrt(1 + tf.square(y))
        tmp = self.A + (self.B - self.A)*self.rho_rev*rho - third
        Q_val = self.alpha * tmp
        return Q_val
    
    def net_f(self, x, t): # physics-informed part
        
        rho = self.net_rho(x,t)
        rho_t = tf.gradients(rho, t)[0]
        
        Q = self.Q_of_rho(rho)
        Q_x = tf.gradients(Q, x)[0]
        
        f = rho_t + Q_x
        
        return f # the residual
        
    
    
    def callback(self, loss):
        #print('Loss: %e, l1: %.5f, l2: %.5f' % (loss, lambda_1, np.exp(lambda_2)))
        print('Loss: %e' % (loss))
        
    
    
    def train(self, nIter):
        tf_dict = {self.x_tf: self.x, self.t_tf: self.t, self.rho_tf: self.rho,    self.x2_tf: self.x2,  self.t2_tf: self.t2}
        
         
        print('dddss')
        print('dddss')
        print('dddss')
        start_time = time.time()
        for it in range(nIter): # why we need to train this??
            self.sess.run(self.train_op_Adam, tf_dict)
            
            # Print
            if it % 10 == 0: # adam training first.. WHY?
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                
                #print('It: %d, Loss: %.3e, Lambda_1: %.3f, Lambda_2: %.6f, Time: %.2f' % (it, loss_value, 1.0, 1.0, elapsed))
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss_value, elapsed))
                start_time = time.time()
        
        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss], # fetch some variables to forward to loss_callback
                                loss_callback = self.callback) # take the value of loss, lambda1 and lambda2 out #this training will continue until epi exceed. for physics problem, maybe L-BFGS is better.
        

        
        


        
    def predict(self, X_star):
        
        tf_dict = {self.x_tf: X_star[:,0:1],  self.t_tf: X_star[:,1:2]}
        
        rho_star = self.sess.run(self.rho_pred, tf_dict)
        #f_star = self.sess.run(self.f_pred, tf_dict)
        
        return rho_star#, f_star









if __name__ == "__main__": 
    
    #nu = 0.01/np.pi
    
    print("lamb",lamb)
    print("p",p)
    print("rho_max", rho_max)
    print("alpha",alpha)
    print("a,b", a,b)
    
    
    N_u = 3000
    #N_u = 1000#20000#100000#50000
    
    layers = [2, 20, 20, 40, 80, 80, 40, 20, 20, 1]
    #layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    #layers = [2, 20, 20, 40, 40, 40, 20, 20, 20, 1]
    
    
    print(data_pickle.keys(), data_pickle['para'].keys())
    print(len(data_pickle['rhoMat']), len(data_pickle['rhoMat'][20]))
    print(len(data_pickle['s']), len(data_pickle['t']))#for pickle data t30s30
    #print(len(data_pickle['x']), len(data_pickle['t']))#for pickle data t10s30
    
    xx = np.array(data_pickle['s'])#for pickle data t30s30
    #xx = np.array(data_pickle['x'])#for pickle data t10s30
    tt = np.array(data_pickle['t'])
    rhoMat = np.array([np.array(ele) for ele in data_pickle['rhoMat']])
    vMat = np.array([np.array(ele) for ele in data_pickle['vMat']])
    
    
    X, T = np.meshgrid(xx,tt)
    print(len(X),len(X[0]))# 21 by 265
    N_u = int(len(X)*len(X[0])*0.8)
    #N_u = 1000
    
    x = X.flatten()[:,None]# 21*265 by 1
    t = T.flatten()[:,None]# 21*265 by 1
    Exact_rho = rhoMat.T # 265 by 21
    
    print(len(t), len(t[0]))
    print(len(x), len(x[0]))
    print(len(Exact_rho), len(Exact_rho[0]))
    
    
    X_star = np.hstack((x, t)) # hstack is column wise stack, 21*265 by 2
    rho_star = Exact_rho.flatten()[:,None] # not 21*265 by 1 => 265*21 by 1
    
    
    
    print(len(X_star),len(X_star[0]), len(rho_star))
    
    # Doman bounds
    lb = X_star.min(0) # [0, 0]
    ub = X_star.max(0) # [1, 3] 
    
    print(lb)
    print(ub)
    
    print(X_star.shape[0])
    
    #N_loop = [0,10,20]
    NN = Loop_Num#22#10#16#2#18
    #N_loop = list(range(0,22, int(22/12)))#[0,6,13,20]
    N_loop = [0]
    tmp = 0
    stepp = 22.0/NN
    for i in range(1,NN):
        tmp+=stepp
        N_loop+=[int(tmp)]
    
    N_loop = N_loop[:-1]+[21]
    print(N_loop)
    #s=sfdsf
    ######################################################################
    ######################## Noiseless Data ###############################
    ######################################################################
    noise = 0.0   
             
    idx = np.random.choice(X_star.shape[0], N_u, replace=False) # N_u = 3000 out of 5565
    idx2 = []
    
    
    for i in range(89):
        base = i*22
        index = [base + ele for ele in N_loop]
        idx2 += index
    
    #idx2 = np.random.choice(range(240), 200, replace=False) # x on the boundary
    
    print(len(rho_star))
    print(len(idx2), len(X_star))
    X_rho_train = X_star[idx2,:] # [x, 0.0] and 100 points from left bound selected
    X_rho_colocat = X_star[idx,:]
    
    
    rho_train = rho_star[idx2,:]
    
    
    print(len(idx),len(idx2))
    #sdf=sdfsfs
    model = PhysicsInformedNN(X_rho_train, X_rho_colocat, rho_train, layers, lb, ub) # the layers is determined at the beginning of the function
    
    
    model.train(4000)
    
    rho_pred = model.predict(X_star)
    
    error_rho = np.linalg.norm(rho_star-rho_pred,2)/np.linalg.norm(rho_star,2)#this is the metric I used
    RHO_pred = griddata(X_star, rho_pred.flatten(), (X, T), method='cubic')
    
    RHO_org = griddata(X_star, Exact_rho.flatten(), (X, T), method='cubic')
    
    
    print('Error rho: %e' % (error_rho))    
    
    
    
    ######################################################################
    ############################# Plotting ###############################
    ######################################################################    
    
    fig, ax = newfig(1.0, 1.4)
    ax.axis('off')
    
    ####### Row 0: u(t,x) ##################    
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1.0/3.0+0.06, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])
    
    h = ax.imshow(RHO_pred.T, interpolation='nearest', cmap='rainbow', 
                  extent=[t.min(), t.max(), x.min(), x.max()], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    #ax.plot(X_rho_train[:,1], X_rho_train[:,0], 'kx', label = 'Data (%d points)' % (rho_train.shape[0]), markersize = 2, clip_on = False)
    
    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)
    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.legend(loc='upper center', bbox_to_anchor=(1.0, -0.125), ncol=5, frameon=False)
    ax.set_title('$rho(t,x)$', fontsize = 10)
    
    ############real##############
    gs2 = gridspec.GridSpec(1, 3)
    gs2.update(top=1.0-2.0/3.0-0.06, bottom=0+0.06, left=0.15, right=0.85, wspace=0.0)
    ax = plt.subplot(gs2[:, :])
    
    h = ax.imshow(RHO_org.T, interpolation='nearest', cmap='rainbow', 
                  extent=[t.min(), t.max(), x.min(), x.max()], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    #ax.plot(X_rho_train[:,1], X_rho_train[:,0], 'kx', label = 'Data (%d points)' % (rho_train.shape[0]), markersize = 2, clip_on = False)
    
    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)
    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.legend(loc='upper center', bbox_to_anchor=(1.0, -0.125), ncol=5, frameon=False)
    ax.set_title('$exact-rho(t,x)$', fontsize = 10)
    
    '''
    ####### Row 1: u(t,x) slices ##################    
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1-1.0/3.0-0.1, bottom=1.0-2.0/3.0, left=0.1, right=0.9, wspace=0.5)
    
    ax = plt.subplot(gs1[0, 0])
    ax.plot(x,Exact[25,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,U_pred[25,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')    
    ax.set_title('$t = 0.25$', fontsize = 10)
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])
    
    ax = plt.subplot(gs1[0, 1])
    ax.plot(x,Exact[50,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,U_pred[50,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])
    ax.set_title('$t = 0.50$', fontsize = 10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)
    
    ax = plt.subplot(gs1[0, 2])
    ax.plot(x,Exact[75,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,U_pred[75,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])    
    ax.set_title('$t = 0.75$', fontsize = 10)
    
    ####### Row 3: Identified PDE ##################    
    gs2 = gridspec.GridSpec(1, 3)
    gs2.update(top=1.0-2.0/3.0, bottom=0, left=0.0, right=1.0, wspace=0.0)
    
    ax = plt.subplot(gs2[:, :])
    ax.axis('off')
    s1 = r'$\begin{tabular}{ |c|c| }  \hline Correct PDE & $u_t + u u_x - 0.0031831 u_{xx} = 0$ \\  \hline Identified PDE (clean data) & '
    s2 = r'$u_t + %.5f u u_x - %.7f u_{xx} = 0$ \\  \hline ' % (lambda_1_value, lambda_2_value)
    s3 = r'Identified PDE (1\% noise) & '
    s4 = r'$u_t + %.5f u u_x - %.7f u_{xx} = 0$  \\  \hline ' % (lambda_1_value_noisy, lambda_2_value_noisy)
    s5 = r'\end{tabular}$'
    s = s1+s2+s3+s4+s5
    ax.text(0.1,0.1,s)
       
    '''
    savefig('./figures/NGSIM_infer') 
    print("number of loops:",Loop_Num)
     
    



