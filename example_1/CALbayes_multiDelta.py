import matplotlib.pyplot as plt
import cPickle
import numpy as np
import scipy.stats as stats
import scipy.io
from scipy.interpolate import interp1d
import time
import random
from lhs import *
from MCMC_multiDelta import *

############################################################################
plt.close('all')
plt.rc('text', usetex=True); plt.rc('font', family='serif')
plt.rc('xtick', labelsize=20); plt.rc('ytick', labelsize=20)

n_trim = 5; n_burnin = 10000; Nmcmc = 100000+n_burnin
n_data = 20; sigma_D = 1.5; x_data = np.linspace(4., 8., n_data)
theta_true = np.array([5.0, 25.0])
theta_2_true = theta_true[0]
theta_3_true = theta_true[1]

y_data = Model_True(x_data, 0.5, theta_2_true, theta_3_true) + scipy.stats.norm.rvs(loc=0.0, scale=sigma_D, size=n_data)
# plt.rc('xtick', labelsize=30); plt.rc('ytick', labelsize=30)
# fig = plt.figure(); ax1 = fig.add_subplot(111)
# ax1.plot(x_data, y_data, 'k*', markersize=20)
# ax1.set_xlabel(r'$x_{Di}$', fontsize=20); ax1.set_ylabel(r'$y_{Di}$', fontsize=20)
# fig_name = 'calibrationData.png'
# fig.savefig(fig_name, dpi=300, facecolor='w', edgecolor='w',
#             orientation='portrait', papertype=None, format='png',
#             transparent=False, bbox_inches='tight', pad_inches=0.1)

data_cal = Data_UQ(y_data, x_data, sigma_D)

params_bnd_0 = np.array([[0.0, 15.0], [1.0, 40.0]])
MCMC_0 = MCMC(params_bnd_0, data_cal, id_delta=0)
MCMC_0.Initial_Search()
MCMC_0.Metropolis(Nmcmc)

params_bnd_1 = np.vstack((params_bnd_0, np.array([0.1, 10.0])))
MCMC_1 = MCMC(params_bnd_1, data_cal, id_delta=1)
MCMC_1.Initial_Search()
MCMC_1.Metropolis(Nmcmc)

params_bnd_2 = np.vstack((params_bnd_0, np.array([[0.1, 5.0], [0.1, 5.0]])))
MCMC_2 = MCMC(params_bnd_2, data_cal, id_delta=2)
MCMC_2.Initial_Search()
MCMC_2.Metropolis(Nmcmc)

params_bnd_3 = np.vstack((params_bnd_0, np.array([[0.1, 5.0], [0.1, 2.0], [0.1, 5.0], [0.1, 5.0]])))
MCMC_3 = MCMC(params_bnd_3, data_cal, id_delta=3)
MCMC_3.Initial_Search()
MCMC_3.Metropolis(Nmcmc)

params_bnd_4 = np.vstack((params_bnd_0, np.array([[0.1, 5.0], [0.1, 2.0], [0.1, 5.0]])))
MCMC_4 = MCMC(params_bnd_4, data_cal, id_delta=4)
MCMC_4.Initial_Search()
MCMC_4.Metropolis(Nmcmc)

MCMC_0.Maximum_Likelihood()
MCMC_1.Maximum_Likelihood()
MCMC_2.Maximum_Likelihood()
MCMC_3.Maximum_Likelihood()
MCMC_4.Maximum_Likelihood()

file = open('result_MCMC_multiDiscrepancy.pickle','wb')
cPickle.dump([MCMC_0, MCMC_1, MCMC_2, MCMC_3, MCMC_4], file)
file.close()

file = open('result_MCMC_multiDiscrepancy.pickle','rb')
MCMC_0, MCMC_1, MCMC_2, MCMC_3, MCMC_4 = cPickle.load(file)
file.close()

MCMC_0.Trim(n_burnin, n_trim)
MCMC_1.Trim(n_burnin, n_trim)
MCMC_2.Trim(n_burnin, n_trim)
MCMC_3.Trim(n_burnin, n_trim)
MCMC_4.Trim(n_burnin, n_trim)

n_params = MCMC_0.n_params
params_bnd = MCMC_0.params_bnd

n_params_grid = 25
params_grid = np.zeros((n_params_grid, n_params))
params_name = [r'$\theta_2$', r'$\theta_3$']
postPDF_params_0 = np.zeros((n_params_grid, n_params))
postPDF_params_1 = np.zeros((n_params_grid, n_params))
postPDF_params_2 = np.zeros((n_params_grid, n_params))
postPDF_params_3 = np.zeros((n_params_grid, n_params))
postPDF_params_4 = np.zeros((n_params_grid, n_params))

plt.rc('text', usetex=True); plt.rc('font', family='serif')
plt.rc('xtick', labelsize=20); plt.rc('ytick', labelsize=20)
for i in xrange(n_params):
    fig = plt.figure(); 
    ax1 = fig.add_subplot(111)
    kernel_0 = stats.gaussian_kde(MCMC_0.params_sample[i, :])
    kernel_1 = stats.gaussian_kde(MCMC_1.params_sample[i, :])
    kernel_2 = stats.gaussian_kde(MCMC_2.params_sample[i, :])
    kernel_3 = stats.gaussian_kde(MCMC_3.params_sample[i, :])
    kernel_4 = stats.gaussian_kde(MCMC_4.params_sample[i, :])
    params_grid[:, i] = np.linspace(params_bnd[i, 0], params_bnd[i, 1], n_params_grid, endpoint=True)
    postPDF_params_0[:, i] = kernel_0(params_grid[:, i])
    postPDF_params_1[:, i] = kernel_1(params_grid[:, i])
    postPDF_params_2[:, i] = kernel_2(params_grid[:, i])
    postPDF_params_3[:, i] = kernel_3(params_grid[:, i])
    postPDF_params_4[:, i] = kernel_4(params_grid[:, i])
    ax1.plot(params_grid[:, i], postPDF_params_0[:, i], linewidth=2.5, label='No delta')
    ax1.plot(params_grid[:, i], postPDF_params_1[:, i], linewidth=2.5, label='delta 1')
    ax1.plot(params_grid[:, i], postPDF_params_2[:, i], linewidth=2.5, label='delta 2')
    ax1.plot(params_grid[:, i], postPDF_params_3[:, i], linewidth=2.5, label='delta 3')
    ax1.plot(params_grid[:, i], postPDF_params_4[:, i], linewidth=2.5, label='delta 4')
    ax1.plot(np.ones(2)*theta_true[i], np.linspace(ax1.get_ylim()[0], ax1.get_ylim()[1], 2), 
             'k--', linewidth=2.5, label='True value')
    ax1.set_xlim(params_bnd[i, 0], params_bnd[i, 1])
    ax1.set_xlabel(params_name[i], fontsize=20)
    ax1.set_ylabel(r'Marginal Posterior', fontsize=20)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles[::1], labels[::1], loc='best', frameon=False)
    fig_name = 'marginalPostPDF_{0}.png'.format(i)
    fig.savefig(fig_name, dpi=300, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format='png',
                transparent=False, bbox_inches='tight', pad_inches=0.1)
    fig_name = 'marginalPostPDF_{0}.pdf'.format(i)
    fig.savefig(fig_name, dpi=300, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format='pdf',
                transparent=False, bbox_inches='tight', pad_inches=0.1)

MCMC_0.Marginal_PDF_Plot()
MCMC_1.Marginal_PDF_Plot()
MCMC_2.Marginal_PDF_Plot()
MCMC_3.Marginal_PDF_Plot()
MCMC_4.Marginal_PDF_Plot()


