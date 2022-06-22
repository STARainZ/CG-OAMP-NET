#!/usr/bin/python

import numpy as np
import numpy.linalg as la
from .utils import NLE

theta = np.ones(20)
gamma = np.ones(20)
beta = np.zeros(20)
phi = np.ones(20)
xi = np.zeros(20)
para = {}

# loading the trained model parameters
try:
    for k, v in np.load("model/rayleigh_qpsk/CG_OAMP_QPSK_8_8_20dB_T4.npz").items():
        para[k] = v
except IOError:
    print("no such file")
    pass
# get parameters for CG-OAMP-NET
for t in range(20):
    if para.get("theta_" + str(t) + ":0", -1) != -1:
        theta[t] = para["theta_" + str(t) + ":0"]
        gamma[t] = para["gamma_" + str(t) + ":0"]
        # beta[t] = para["beta_"+str(t)+":0"]
        phi[t] = para["phi_" + str(t) + ":0"]
        xi[t] = para["xi_" + str(t) + ":0"]


def CG_OAMP(x, A, y, noise_var, T=5, I=50, mu=2):
    # initialize
    M = A.shape[0]
    N = A.shape[1]
    v_sqr_last = 0.
    xt = np.zeros((N, 1))
    AH = A.T
    AAH = A @ AH
    C = AAH[:M // 2, :M // 2] + 1j * AAH[M // 2:M, :M // 2]
    eigvalue = la.eigvalsh(C)
    MSE = np.zeros(T, dtype=np.float64)
    Iter = np.zeros(T, dtype=int)

    for t in range(T):
        ''' LE '''
        p_noise = y - A @ xt
        v_sqr = (np.square(np.linalg.norm(p_noise, 2, axis=0)) - M * noise_var) / np.trace(AAH)
        sigmoid_beta = 1. / (1. + np.exp(-beta[t]))  # damping factor, default is 0.5
        v_sqr = sigmoid_beta * v_sqr + (1 - sigmoid_beta) * v_sqr_last  # damping
        v_sqr = np.maximum(v_sqr, 1e-10)  # in case that v_sqr is negative
        v_sqr_last = v_sqr

        # CG for the calculation of u = (xi)**-1 @ p_noise
        XI = AAH + noise_var / v_sqr * np.eye(M)
        u = np.zeros_like(y)
        residual = p_noise
        p = residual
        r_norm = residual.T @ residual
        for i in range(I):
            # compute the approximate solution based on prior conjugate direction and residual
            xi_p = XI @ p
            a = r_norm / (p.T @ xi_p)
            u += a * p
            # compute conjugate direction and residual
            r_norm_last = r_norm
            residual = residual - a * xi_p
            r_norm = residual.T @ residual
            b = r_norm / r_norm_last
            p = residual + b * p
            if r_norm < 1e-8:
                # print(i)
                break
        Iter[t] = i + 1

        estimate = sum(eigvalue / (eigvalue + noise_var / v_sqr)) * 2
        nor_coef = N / estimate
        tau_sqr = v_sqr * ((theta[t] ** 2) * nor_coef - 2 * theta[t] + 1)
        tau_sqr = np.maximum(tau_sqr, 1e-10)
        r = xt + gamma[t] * nor_coef * AH @ u

        ''' NLE '''
        xhat, vhat, xt, v_sqr = NLE(tau_sqr, r, orth=False, mu=mu)
        MSE[t] = np.mean((x - xhat) ** 2)
        xt = phi[t] * (xhat - xi[t] * r)

    return xt, Iter, MSE
