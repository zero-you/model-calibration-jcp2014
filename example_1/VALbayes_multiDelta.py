import matplotlib.pyplot as plt
import cPickle
import numpy as np
#from numpy import *
import scipy.stats as stats
import scipy.io
from scipy.interpolate import interp1d
import time
import random
import copy
from lhs import *
from MCMC_multiDelta import *

############################################################################
def Uncertainty_Propagation(M):
    y_m = np.zeros((M.data.y_D.size, M.params_sample.shape[1]))
    for i in xrange(M.data.y_D.size):
        y_m[i,:] = Model_Test(M.data.x_D[i], M.params_sample[0,:], M.params_sample[1,:])
    return y_m

class Probability_Bounds:
    def __init__(self, y, p1, p2):
        self.y = y
        self.p1 = p1
        self.p2 = p2
        y_temp = np.sort(y, axis=1)
        self.y_lb = y_temp[:, int(p1*y.shape[1])]
        self.y_ub = y_temp[:, int(p2*y.shape[1])]
        self.y_mean = np.mean(y, axis=1)        

    def Plot(self, data, line_label):
        x_D = data.x_D
        y_D = data.y_D
        y_mean = self.y_mean
        y_lb = self.y_lb
        y_ub = self.y_ub
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.plot(x_D, y_mean, linewidth=2.5, label=line_label[0])
        ax1.plot(x_D, y_lb, '--', linewidth=2.5, label='')
        ax1.plot(x_D, y_ub, '--', linewidth=2.5, label='')
        ax1.plot(x_D, y_D, 'k*', linewidth=2.5, label=line_label[1])
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles[::1], labels[::1], loc='best', frameon=False)
        self.fig = fig1

    def Save_Figure(self, fig_name):        
        self.fig_name = fig_name
        self.fig.savefig(fig_name+'.png', dpi=300, facecolor='w', edgecolor='w',
                     orientation='portrait', papertype=None, format='png',
                     transparent=False, bbox_inches='tight', pad_inches=0.1)
        self.fig.savefig(fig_name+'.pdf', dpi=300, facecolor='w', edgecolor='w',
                     orientation='portrait', papertype=None, format='pdf',
                     transparent=False, bbox_inches='tight', pad_inches=0.1)


def Delta_0(M):
    Nmcmc = M.params_sample.shape[1]
    n_data = M.data.x_D.shape[0]
    delta_sample = np.zeros((n_data, Nmcmc))
    for i in xrange(n_data):
        delta_sample[i, :] = scipy.stats.norm.rvs(loc=0.,scale=M.data.sigma_D,size=Nmcmc)
    return delta_sample

def Delta_1(M):
    Nmcmc = M.params_sample.shape[1]
    n_data = M.data.x_D.shape[0]
    delta_sample = np.zeros((n_data, Nmcmc))
    for i in xrange(n_data):
        delta_sample[i,:] = M.params_sample[2, :]+scipy.stats.norm.rvs(loc=0.,scale=M.data.sigma_D,size=Nmcmc)
    return delta_sample

def Delta_2(M):
    Nmcmc = M.params_sample.shape[1]
    mean_delta = M.params_sample[2, :]
    std_delta = M.params_sample[3, :]
    s = scipy.stats.norm.rvs(loc=0.,scale=1.,size=Nmcmc)
    n_data = M.data.x_D.shape[0]
    delta_sample = np.zeros((n_data, Nmcmc))
    for i in xrange(n_data):
        delta_sample[i,:] = s*(std_delta+M.data.sigma_D)+mean_delta
    return delta_sample, mean_delta, std_delta

def Delta_3(M):
    n_data = M.data.x_D.shape[0]
    Nmcmc = M.params_sample.shape[1]
    mean_delta = np.zeros((n_data,Nmcmc))
    std_delta = np.zeros((n_data,Nmcmc))
    delta_sample = np.zeros((n_data,Nmcmc))
    for i in xrange(n_data):
        mean_delta[i, :] = M.params_sample[2, :]*M.data.x_D[i]**2
        std_delta[i, :] = M.params_sample[4, :]*M.data.x_D[i]**2 + M.params_sample[3, :]*M.data.x_D[i] + M.params_sample[5, :]
        s = scipy.stats.norm.rvs(loc=0.,scale=1.,size=Nmcmc)
        delta_sample[i, :] = s*(std_delta[i, :]+M.data.sigma_D)+mean_delta[i,:]
    return delta_sample, mean_delta, std_delta

def Delta_4(M):
    n_data = M.data.x_D.shape[0]
    Nmcmc = M.params_sample.shape[1]
    mean_delta = np.zeros((n_data,Nmcmc))
    for i in xrange(n_data):
        mean_delta[i, :] = M.params_sample[2, :]*M.data.x_D[i]**2
    delta_sample = np.zeros((n_data, Nmcmc))
    for k in xrange(Nmcmc):
        if np.mod(k, 1000)==0:
            print k
        cov_data = np.zeros((n_data, n_data))
        for i in xrange(n_data):
            for j in xrange(n_data):
                if i<=j:
                    cov_data[i, j] = M.params_sample[3, k]*np.exp(-0.5*((M.data.x_D[i]-M.data.x_D[j])/M.params_sample[4, k])**2)
                else:
                    cov_data[i, j] = cov_data[j, i]
                if i==j:
                    cov_data[i, j] = cov_data[i, j] + M.data.sigma_D**2
        sigma_delta = np.sqrt(np.diag(cov_data))
        corr_data = np.zeros((n_data, n_data))
        for i in xrange(n_data):
            for j in xrange(n_data):
                if i<=j:
                    corr_data[i, j] = cov_data[i, j]/(sigma_delta[i]*sigma_delta[j])
                else:
                    corr_data[i, j] = corr_data[j, i]                    
        L_chol = np.linalg.cholesky(corr_data)
        w = scipy.stats.norm.rvs(loc=0.,scale=1.,size=n_data)
        s = np.linalg.solve(L_chol, w)
        delta_sample[:, k] = s*sigma_delta+mean_delta[:, k]
    return delta_sample, mean_delta


class Graphical_Validation:
    def __init__(self, M_list, y, domain='CAL'):
        self.y = y
        self.M_list = M_list
        self.domain = domain
        self.M_list_len = len(M_list)
        self.x_D = range(self.M_list_len)
        self.y_D = range(self.M_list_len)
        self.y_mean = range(self.M_list_len)
        self.y_lb = range(self.M_list_len)
        self.y_ub = range(self.M_list_len)
        for i in xrange(self.M_list_len):
            self.x_D[i] = M_list[i].data.x_D
            self.y_D[i] = M_list[i].data.y_D
            y_pb = Probability_Bounds(y[i], 0.05, 0.95)
            self.y_mean[i], self.y_lb[i], self.y_ub[i] = y_pb.y_mean, y_pb.y_lb, y_pb.y_ub

    def Plot(self):
        M_list = self.M_list
        M_list_len = self.M_list_len
        x_D = self.x_D
        y_D = self.y_D
        y_mean = self.y_mean
        y_lb = self.y_lb
        y_ub = self.y_ub
        fig1 = plt.figure(figsize=(12.125, 6.125))
        ax1 = range(M_list_len)
        for i in xrange(M_list_len):
            ax1[i] = fig1.add_subplot(2,3,i+1)
            ax1[i].plot(x_D[i], y_mean[i], linewidth=2.5, label='delta {0}'.format(i))
            ax1[i].plot(x_D[i], y_lb[i], '--', linewidth=2.5, label='')
            ax1[i].plot(x_D[i], y_ub[i], '--', linewidth=2.5, label='')
            ax1[i].plot(x_D[i], y_D[i], 'k*', linewidth=2.5, label='Data')
            handles, labels = ax1[i].get_legend_handles_labels()
            ax1[i].legend(handles[::1], labels[::1], loc='best', frameon=False)            
        self.fig = fig1

    def Save_Figure(self, fig_name):
        self.fig.savefig(fig_name+'.png', dpi=300, facecolor='w', edgecolor='w',
                         orientation='portrait', papertype=None, format='png',
                         transparent=False, bbox_inches='tight', pad_inches=0.1)
        self.fig.savefig(fig_name+'.pdf', dpi=300, facecolor='w', edgecolor='w',
                         orientation='portrait', papertype=None, format='pdf',
                         transparent=False, bbox_inches='tight', pad_inches=0.1)

class Reliability_Metric:
    def __init__(self, data, model, tol):
        self.data = data
        self.model = model
        self.tol = tol
        self.n_data = data.y_D.size
        self.Nmcmc = model.shape[1]
        self.sample = np.zeros(self.n_data)
        for i in xrange(self.n_data):
            d = np.abs(model[i, :]-data.y_D[i])
            self.sample[i] = float(d[d<tol].size)/float(self.Nmcmc)
        self.mean = np.mean(self.sample)
    
    def Histogram(self, ax1, id_delta, n_grid=25):
        #fig = plt.figure(); 
        #ax1 = fig.add_subplot(111)
        kernel = stats.gaussian_kde(self.sample)
        grid = np.linspace(0.0, 1.0, n_grid, endpoint=True)
        hist_rm = kernel(grid)
        ax1.plot(grid, hist_rm, linewidth=2.5, label='delta {0}'.format(id_delta))
        ax1.set_xlim(0, 1)
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles[::1], labels[::1], loc='best', frameon=False)

class Bayesian_Model_Averaging:
    def __init__(self, pm, sample, Nmc):
        self.pm = pm
        self.Nmc = Nmc
        self.Nmc_individual = np.round(Nmc*pm)
        self.sample = sample

    def Sampling(self):
        sample = self.sample
        self.random_id = range(len(sample))
        self.sample_bma_individual = range(len(sample))
        self.sample_bma = np.empty((sample[0].shape[0], 0))
        for i in xrange(len(sample)):
            random_id = np.random.randint(0, sample[i].shape[1], self.Nmc_individual[i])
            self.sample_bma_individual[i] = sample[i][:, random_id]
            self.sample_bma = np.hstack((self.sample_bma, self.sample_bma_individual[i]))

############################################################################
plt.close('all')
plt.rc('xtick', labelsize=10); plt.rc('ytick', labelsize=10)
plt.rc('font', size=10)

n_trim = 5; n_burnin = 5000; #Nmcmc = 100000+n_burnin
n_data = 20; sigma_D = 1.5; x_data = np.linspace(8.5, 12.5, n_data)
theta_true = np.array([5.0, 25.0])
theta_2_true = theta_true[0]
theta_3_true = theta_true[1]

y_data = Model_True(x_data, 0.5, theta_2_true, theta_3_true) + scipy.stats.norm.rvs(loc=0.0, scale=sigma_D, size=n_data)
# fig = plt.figure(); ax1 = fig.add_subplot(111)
# ax1.plot(x_data, y_data, 'k*', markersize=20)
# ax1.set_xlabel(r'$x_{Di}$', fontsize=20); ax1.set_ylabel(r'$y_{Di}$', fontsize=20)
data_val = Data_UQ(y_data, x_data, sigma_D)

file = open('result_MCMC_multiDiscrepancy.pickle','rb')
MCMC_0, MCMC_1, MCMC_2, MCMC_3, MCMC_4 = cPickle.load(file)
file.close()

print MCMC_0.maxjlk
print MCMC_1.maxjlk
print MCMC_2.maxjlk
print MCMC_3.maxjlk
print MCMC_4.maxjlk

MCMC_0.Trim(n_burnin, n_trim)
MCMC_1.Trim(n_burnin, n_trim)
MCMC_2.Trim(n_burnin, n_trim)
MCMC_3.Trim(n_burnin, n_trim)
MCMC_4.Trim(n_burnin, n_trim)

# MCMC_0.data = data_val
# MCMC_1.data = data_val
# MCMC_2.data = data_val
# MCMC_3.data = data_val
# MCMC_4.data = data_val
# domain_name = 'VAL'
domain_name = 'CAL'

y_m0 = Uncertainty_Propagation(MCMC_0)
y_m1 = Uncertainty_Propagation(MCMC_1)
y_m2 = Uncertainty_Propagation(MCMC_2)
y_m3 = Uncertainty_Propagation(MCMC_3)
y_m4 = Uncertainty_Propagation(MCMC_4)

delta_0_sample = Delta_0(MCMC_0)
delta_1_sample = Delta_1(MCMC_1)
delta_2_sample, mean_delta_2, std_delta_2 = Delta_2(MCMC_2)
delta_3_sample, mean_delta_3, std_delta_3 = Delta_3(MCMC_3)
delta_4_sample, mean_delta_4 = Delta_4(MCMC_4)

y_0 = y_m0 + delta_0_sample
y_1 = y_m1 + delta_1_sample
y_2 = y_m2 + delta_2_sample
y_3 = y_m3 + delta_3_sample
y_4 = y_m4 + delta_4_sample

# Graphical validation
M_list = [MCMC_0, MCMC_1, MCMC_2, MCMC_3, MCMC_4]
y_list0 = [y_m0, y_m1, y_m2, y_m3, y_m4]
y_list1 = [y_0, y_1, y_2, y_3, y_4]
GV0 = Graphical_Validation(M_list, y_list0, domain=domain_name)
GV1 = Graphical_Validation(M_list, y_list1, domain=domain_name)
GV0.Plot()
GV1.Plot()
fig_name = ['PredictionWithoutDelta_'+domain_name, 'PredictionWithDelta_'+domain_name]
GV0.Save_Figure(fig_name=fig_name[0])
GV1.Save_Figure(fig_name=fig_name[1])

if domain_name=='VAL':
    # Reliability-based metric
    tol = 4.5
    rm_0 = Reliability_Metric(data_val, y_0, tol)
    rm_1 = Reliability_Metric(data_val, y_1, tol)
    rm_2 = Reliability_Metric(data_val, y_2, tol)
    rm_3 = Reliability_Metric(data_val, y_3, tol)
    rm_4 = Reliability_Metric(data_val, y_4, tol)
    print rm_0.mean, rm_1.mean, rm_2.mean, rm_3.mean, rm_4.mean

    mean_rm = np.array([rm_0.mean, rm_1.mean, rm_2.mean, rm_3.mean, rm_4.mean])
    mean_rm = mean_rm/np.sum(mean_rm)

    theta_sample = [MCMC_0.params_sample[0:2,:], MCMC_1.params_sample[0:2,:], MCMC_2.params_sample[0:2,:], MCMC_3.params_sample[0:2,:], MCMC_4.params_sample[0:2,:]]
    delta_sample = [delta_0_sample, delta_1_sample, delta_2_sample, delta_3_sample, delta_4_sample]
    theta_bma = Bayesian_Model_Averaging(mean_rm, theta_sample, 10000)
    theta_bma.Sampling()
    theta_sample_bma = theta_bma.sample_bma
    delta_bma = Bayesian_Model_Averaging(mean_rm, delta_sample, 10000)
    delta_bma.Sampling()
    delta_sample_bma = delta_bma.sample_bma

    M_bma = copy.deepcopy(MCMC_0)
    M_bma.params_sample = theta_sample_bma
    M_bma.id_delta = 'bma'
    M_bma.Marginal_PDF_Plot()

    # y_bma = Uncertainty_Propagation(M_bma)
    # fig_name = 'Prediction_BMA'
    y_bma = Uncertainty_Propagation(M_bma) + delta_sample_bma
    fig_name = 'Prediction_BMAwithDelta'
    
    y_bma_pb = Probability_Bounds(y_bma, 0.05, 0.95)
    line_label = ['BMA', 'Data']
    y_bma_pb.Plot(MCMC_0.data, line_label)
    y_bma_pb.Save_Figure(fig_name)

    fig4 = plt.figure()
    ax1 = fig4.add_subplot(111)
    rm_0.Histogram(ax1, 0)
    rm_1.Histogram(ax1, 1)
    rm_2.Histogram(ax1, 2)
    rm_3.Histogram(ax1, 3)
    rm_4.Histogram(ax1, 4)
    fig_name = 'histogram_rm.png'
    fig4.savefig(fig_name, dpi=300, facecolor='w', edgecolor='w',
                 orientation='portrait', papertype='tight', format='png',
                 transparent=False, bbox_inches='tight', pad_inches=0.1)
    fig_name = 'histogram_rm.pdf'
    fig4.savefig(fig_name, dpi=300, facecolor='w', edgecolor='w',
                 orientation='portrait', papertype=None, format='pdf',
                 transparent=False, bbox_inches='tight', pad_inches=0.1)


