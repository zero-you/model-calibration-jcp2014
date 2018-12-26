import numpy as np
import scipy.stats

############################################################################
def Model_True(x, theta_1, theta_2, theta_3):
    y = theta_1*x**2 + theta_2*x + theta_3
    return y

def Model_Test(x, theta_2, theta_3):
    y = theta_2*x + theta_3
    return y

# No discrepancy
def Log_Joint_Likelihood_0(params, data):
    y_D = data.y_D
    sigma_D = data.sigma_D
    x_D = data.x_D
    n_data = x_D.shape[0]
    theta_2 = params[0]
    theta_3 = params[1]
    y_m = Model_Test(x_D, theta_2, theta_3)
    log_jlk_params = np.sum(np.log(scipy.stats.norm.pdf(y_D, loc=y_m, scale=sigma_D, size=n_data)))
    return log_jlk_params

# Discrepancy as an unknown constant
def Log_Joint_Likelihood_1(params, data):
    y_D = data.y_D
    sigma_D = data.sigma_D
    x_D = data.x_D
    n_data = x_D.shape[0]
    theta_2 = params[0]
    theta_3 = params[1]
    delta = params[2]
    y_m = Model_Test(x_D, theta_2, theta_3)
    log_jlk_params = np.sum(np.log(scipy.stats.norm.pdf(y_D, loc=y_m+delta, scale=sigma_D, size=n_data)))
    return log_jlk_params

# Discrepancy with constant (but unknown) mean and variance
def Log_Joint_Likelihood_2(params, data):
    y_D = data.y_D
    sigma_D = data.sigma_D
    x_D = data.x_D
    n_data = x_D.shape[0]
    theta_2 = params[0]
    theta_3 = params[1]
    mean_delta = params[2]
    std_delta = params[3]
    y_m = Model_Test(x_D, theta_2, theta_3)
    log_jlk_params = np.sum(np.log(scipy.stats.norm.pdf(y_D, loc=y_m+mean_delta, scale=np.sqrt(std_delta**2+sigma_D**2), size=n_data)))
    return log_jlk_params

# # Discrepancy with zero mean and known variance
# def Log_Joint_Likelihood_2(params, data):
#     y_D = data.y_D
#     sigma_D = data.sigma_D
#     x_D = data.x_D
#     n_data = x_D.shape[0]
#     theta_2 = params[0]
#     theta_3 = params[1]
#     std_delta = 10.0
#     y_m = Model_Test(x_D, theta_2, theta_3)
#     log_jlk_params = np.sum(np.log(scipy.stats.norm.pdf(y_D, loc=y_m, scale=sqrt(std_delta**2+sigma_D**2), size=n_data)))
#     return log_jlk_params

# Gaussian random variable with input-dependent mean and variance
def Log_Joint_Likelihood_3(params, data):
    y_D = data.y_D
    sigma_D = data.sigma_D
    x_D = data.x_D
    n_data = x_D.shape[0]
    theta_2 = params[0]
    theta_3 = params[1]
    mean_delta = params[2]*x_D**2
    std_delta = params[3]*x_D**2 + params[4]*x_D + params[5] # how to make sure std_delta is always positive?
    y_m = Model_Test(x_D, theta_2, theta_3)
    log_jlk_params = np.sum(np.log(scipy.stats.norm.pdf(y_D, loc=y_m+mean_delta, scale=np.sqrt(std_delta**2+sigma_D**2), size=n_data)))
    return log_jlk_params

# Gaussian process discrepancy
def Log_Joint_Likelihood_4(params, data):
    y_D = data.y_D
    sigma_D = data.sigma_D
    x_D = data.x_D
    n_data = x_D.shape[0]
    theta_2 = params[0]
    theta_3 = params[1]
    mean_delta = params[2]*x_D**2
    y_m = Model_Test(x_D, theta_2, theta_3)
    h = y_D-y_m-mean_delta

    cov_data = np.zeros((n_data, n_data))
    for i in xrange(n_data):
        for j in xrange(n_data):
            if i<=j:
                cov_data[i, j] = params[3]*np.exp(-0.5*((x_D[i]-x_D[j])/params[4])**2)
            else:
                cov_data[i, j] = cov_data[j, i]
            if i==j:
                cov_data[i, j] = cov_data[i, j] + sigma_D**2

    try:
        L_chol = np.linalg.cholesky(cov_data)
    except np.linalg.LinAlgError:
        L_chol = None
    if L_chol is not None:
        temp_1 = np.linalg.solve(L_chol, h)
        temp_2 = np.linalg.solve(L_chol.T, temp_1)
        log_jlk_params = -np.sum(np.log(np.diag(L_chol))) - 0.5*np.dot(h, temp_2)
    else:
        log_jlk_params = -np.inf
    return log_jlk_params

def Log_Joint_Likelihood_4s(params, data):
    y_D = data.y_D
    sigma_D = data.sigma_D
    x_D = data.x_D
    n_data = x_D.shape[0]
    y_D_b = data.y_D_b
    x_D_b = data.x_D_b
    index_b = data.index_b
    n_data_b = x_D_b.shape[0]

    theta_2 = params[0]
    theta_3 = params[1]
    mean_delta = params[2]*x_D**2
    y_m = Model_Test(x_D, theta_2, theta_3)
    h = y_D-y_m-mean_delta

    K_nn = np.diag(np.ones(n_data)*params[3])
    K_M = np.zeros((n_data_b, n_data_b))
    K_NM = np.zeros((n_data, n_data_b))
    for i in xrange(n_data):
        for j in xrange(n_data_b):
            K_NM[i,j] = params[3]*np.exp(-0.5*((x_D[i]-x_D_b[j])/params[4])**2)
    for i in xrange(n_data_b):
        for j in xrange(n_data_b):
            if i<=j:
                K_M[i,j] = params[3]*np.exp(-0.5*((x_D_b[i]-x_D_b[j])/params[4])**2)
            else:
                K_M[i,j] = K_M[j,i]
    try:
        L_chol_M = np.linalg.cholesky(K_M)
    except np.linalg.LinAlgError:
        L_chol_M = None
    if L_chol_M is not None:
        temp_1 = np.linalg.solve(L_chol_M,K_NM.T)
        temp_2 = np.linalg.solve(L_chol_M.T, temp_1)
        temp_3 = np.dot(K_NM,temp_2)
        Lambda = np.diag(K_nn) - np.diag(np.diag(temp_3))
        cov_delta = temp_3 + Lambda + sigma_D*np.eye(n_data)
        try:
            L_chol = np.linalg.cholesky(cov_delta)
        except np.linalg.LinAlgError:
            L_chol = None
        if L_chol is not None:
            temp_1 = np.linalg.solve(L_chol, h)
            temp_2 = np.linalg.solve(L_chol.T, temp_1)
            log_jlk_params = -np.sum(np.log(np.diag(L_chol))) - 0.5*np.dot(h, temp_2)
        else:
            log_jlk_params = -np.inf
    else:
        log_jlk_params = -np.inf
    return log_jlk_params
