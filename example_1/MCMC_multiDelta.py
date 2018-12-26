import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
#import scipy.io
#from scipy.interpolate import interp1d
#import random
from lhs import *
from CALbayes_Models import *

############################################################################
class Data_UQ:
    def __init__(self, y_D, x_D, sigma_D, sparse=0, y_D_b=None, x_D_b=None, index_b=None):
        self.y_D = y_D
        self.x_D = x_D
        self.sigma_D = sigma_D
        self.sparse = sparse
        if (y_D_b is not None) & (x_D_b is not None) & (index_b is not None):
            self.y_D_b = y_D_b
            self.x_D_b = x_D_b
            self.index_b = index_b
    
class MCMC:
    def __init__(self, params_bnd, data, params_init=None, id_delta=0):
        self.step_size = 1./30.
        self.data = data
        self.params_bnd = params_bnd
        self.n_params = params_bnd.shape[0]
        self.params_init = params_init
        self.id_delta = id_delta
        self.params_sample = []
        self.a = []
        self.log_jlk_params_MCMC = []
        if id_delta==0:
            self.Log_Joint_Likelihood = Log_Joint_Likelihood_0
        elif id_delta==1:
            self.Log_Joint_Likelihood = Log_Joint_Likelihood_1
        elif id_delta==2:
            self.Log_Joint_Likelihood = Log_Joint_Likelihood_2
        elif id_delta==3:
            self.Log_Joint_Likelihood = Log_Joint_Likelihood_3
        elif id_delta==4:
            if data.sparse==0:
                self.Log_Joint_Likelihood = Log_Joint_Likelihood_4
            elif data.sparse==1:
                self.Log_Joint_Likelihood = Log_Joint_Likelihood_4s

    def Initial_Search(self, n_lhsample=500, log_jlk_params_min=-300):
        data = self.data
        n_params = self.n_params
        params_bnd = self.params_bnd
        dist = [stats.uniform]
        for i in xrange(self.n_params-1):
            dist.append(stats.uniform)
        dist = tuple(dist)
        pars = np.array([np.zeros(n_params)])
        pars = tuple(tuple(x) for x in pars.T)    
        log_jlk_params_init = log_jlk_params_min - 1
        while log_jlk_params_init<log_jlk_params_min:
            lhsample_temp =  lhs(dist, pars, siz=n_lhsample)
            lhsample = np.zeros((n_lhsample, n_params))
            log_jlk_params_init_temp = np.zeros(n_lhsample)
            for i in xrange(n_params):
                if n_params>1:
                    lhsample[:, i] = params_bnd[i, 0] + lhsample_temp[i]*(params_bnd[i, 1]-params_bnd[i, 0]) 
                else:
                    lhsample[:, i] = params_bnd[i, 0] + lhsample_temp*(params_bnd[i, 1]-params_bnd[i, 0]) 
            for i in xrange(n_lhsample):
                log_jlk_params_init_temp[i] = self.Log_Joint_Likelihood(lhsample[i, :], data)
                #print log_jlk_params_init_temp[i]
            id_max = log_jlk_params_init_temp.argmax()
            params_init = lhsample[id_max, :]
            log_jlk_params_init = log_jlk_params_init_temp[id_max]
        self.params_init = params_init

    def Metropolis(self, Nmcmc=1000):
        data = self.data
        n_params = self.n_params
        params_bnd = self.params_bnd
        step_size = self.step_size
        params_init = self.params_init
        log_jlk_params_init = self.Log_Joint_Likelihood(params_init, data)
        params_old = params_init.copy()
        log_jlk_params_old = log_jlk_params_init
        alpha = scipy.stats.distributions.uniform.rvs(0, 1, size=Nmcmc)
        params_sample = np.zeros((n_params, Nmcmc))        
        a = np.zeros(Nmcmc)
        log_jlk_params_MCMC = np.zeros(Nmcmc)
        print 'progress: '
        for i in xrange(Nmcmc):
            if_in_bound = 0
            log_jlk_params_new = -np.inf
            while not(if_in_bound) or log_jlk_params_new==-np.inf:
                params_new = params_old + (np.random.random_sample(n_params)*2.0-1.0)*(params_bnd[:, 1]-params_bnd[:, 0])*step_size
                if_in_bound = np.prod((params_new>=params_bnd[:,0])*(params_new<=params_bnd[:,1]))
                log_jlk_params_new = self.Log_Joint_Likelihood(params_new, data)
            
            a[i] = min(np.exp(log_jlk_params_new-log_jlk_params_old), 1.0)
            if alpha[i]<=a[i]:
                params_old = params_new
                params_sample[:, i] = params_new.copy()
                log_jlk_params_old = log_jlk_params_new
            else:
                params_sample[:, i] = params_old.copy()
            log_jlk_params_MCMC[i] = log_jlk_params_old
            if np.mod(i-1, int(Nmcmc/100.0))==0:
                print (float(i)/float(Nmcmc)*100), '%'
                a_temp = a[0:i].copy(); a_temp[a_temp>1]=1
                if np.mean(a_temp)<0.1:
                    step_size = step_size/2.0
                if np.mean(a_temp)>0.35:
                    step_size = step_size*2.0        
                self.step_size = step_size
        self.params_sample = params_sample
        self.a = a
        self.log_jlk_params_MCMC = log_jlk_params_MCMC
        print 'Done!\n'

    def Maximum_Likelihood(self):
        self.maxjlk = self.params_sample[:, self.log_jlk_params_MCMC.argmax()]

    def Trim(self, n_burnin, n_trim):
        self.params_sample = self.params_sample[:, n_burnin::n_trim] 
        self.log_jlk_params_MCMC = self.log_jlk_params_MCMC[n_burnin::n_trim] 
        self.a = self.a[n_burnin::n_trim] 

    def Marginal_PDF_Plot(self, n_params_grid=25, savefig=0):
        n_params = self.n_params
        params_bnd = self.params_bnd
        params_grid = np.zeros((n_params_grid, n_params))
        postPDF_params = np.zeros((n_params_grid, n_params))
        for i in xrange(n_params):
            fig = plt.figure(); 
            ax1 = fig.add_subplot(111)
            kernel = stats.gaussian_kde(self.params_sample[i, :])
            params_grid[:, i] = np.linspace(params_bnd[i, 0], params_bnd[i, 1], n_params_grid, endpoint=True)
            postPDF_params[:, i] = kernel(params_grid[:, i])
            ax1.plot(params_grid[:, i], postPDF_params[:, i], linewidth=2.5, label='')
            ax1.set_xlim(params_bnd[i, 0], params_bnd[i, 1])
            #ax1.set_xlabel(params_name[i], fontsize=20)
            ax1.set_ylabel(r'Marginal Posterior', fontsize=20)
            handles, labels = ax1.get_legend_handles_labels()
            ax1.legend(handles[::1], labels[::1], loc='best', frameon=False)
            if savefig==1:
                fig_name = 'marginalPostPDF_delta_{0}_{1}'.format(self.id_delta,i)
                fig.savefig(fig_name+'.png', dpi=300, facecolor='w', edgecolor='w',
                            orientation='portrait', papertype='tight', format='png',
                            transparent=False, bbox_inches='tight', pad_inches=0.1)
                fig.savefig(fig_name+'.pdf', dpi=300, facecolor='w', edgecolor='w',
                            orientation='portrait', papertype=None, format='pdf',
                            transparent=False, bbox_inches='tight', pad_inches=0.1)
