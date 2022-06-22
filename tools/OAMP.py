#!/usr/bin/python

import numpy as np
import numpy.linalg as la
from .utils import NLE

theta = np.ones(20)
gamma = np.ones(20)
phi = np.ones(20)
xi = np.zeros(20)
para = {}

# loading the trained model parameters
try:
    for k, v in np.load("model/rayleigh_qpsk/OAMP_QPSK_8_8_20dB_T4.npz").items():
        para[k] = v
except IOError:
    print("no such file")
    pass
# get parameters for OAMP-NET
for t in range(20):
    if para.get("theta_" + str(t) + ":0", -1) != -1:
        theta[t] = para["theta_" + str(t) + ":0"]
        gamma[t] = para["gamma_" + str(t) + ":0"]
        phi[t] = para["phi_" + str(t) + ":0"]
        xi[t] = para["xi_" + str(t) + ":0"]


def OAMP(x, A, y, noise_var, T=5, mu=2):
    # initialize
    M = A.shape[0]
    N = A.shape[1]
    v_sqr_last = 0.
    # v_sqr = 1.
    xt = np.zeros((N, 1))
    AH = A.T
    AAH = A @ AH
    MSE = np.zeros(T, dtype=np.float64)
    for t in range(T):
        ''' LE '''
        p_noise = y - A @ xt
        v_sqr = (np.square(np.linalg.norm(p_noise, 2, axis=0)) - M * noise_var) / np.trace(AAH)
        v_sqr = 0.5 * v_sqr + 0.5 * v_sqr_last
        v_sqr = np.maximum(v_sqr, 1e-10)  # in case that v_sqr is negative
        v_sqr_last = v_sqr

        w_hat = AH @ np.linalg.inv(AAH + noise_var * np.eye(M) / v_sqr)
        nor_coef = N / np.trace(w_hat @ A)
        r = xt + gamma[t] * nor_coef * w_hat @ p_noise

        tau_sqr = v_sqr * ((theta[t] ** 2) * nor_coef - 2 * theta[t] + 1)
        tau_sqr = np.maximum(tau_sqr, 1e-10)
        ''' NLE '''
        xhat, vhat, xt, v_sqr = NLE(tau_sqr, r, orth=False, mu=mu)  # what if directly use v_post=vhat
        MSE[t] = np.mean((x - xhat) ** 2)
        xt = phi[t] * (xhat - xi[t] * r)

    return xhat, MSE
