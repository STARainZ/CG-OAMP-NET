#!/usr/bin/python
from __future__ import division
import numpy as np
import time
import sys
import math
import numpy.linalg as la
from scipy.linalg import toeplitz, sqrtm
from .utils import QAM_Modulation, QAM_Demodulation
from .OAMP import OAMP
from .CG_OAMP import CG_OAMP
from .MAMP import MAMP

pi = math.pi


def MIMO_detection_simulate(sysin, SNR=40):
    Mr, Nt, mu = sysin.Mr, sysin.Nt, sysin.mu
    channel_type, rho_tx, rho_rx = sysin.channel_type, sysin.rho_tx, sysin.rho_rx
    err_bits_target = 1000
    total_err_bits = 0
    total_bits = 0
    count = 0
    start = time.time()
    MSE = 0
    total_time = 0.
    if channel_type == 'corr':
        sqrtRtx, sqrtRrx = corr_channel(Mr, Nt, rho_tx=rho_tx, rho_rx=rho_rx)

    while True:
        # generate bits and modulate
        bits = np.random.binomial(n=1, p=0.5, size=(Nt * mu,))  # label
        bits_mod = QAM_Modulation(bits, mu)
        x = bits_mod.reshape(Nt, 1)

        H = np.sqrt(1 / 2 / Mr) * (np.random.randn(Mr, Nt)
                                   + 1j * np.random.randn(Mr, Nt))  # Rayleigh MIMO channel
        if channel_type == 'corr':  # Correlated MIMO channel
            H = sqrtRrx @ H @ sqrtRtx

        # channel input & output
        y = H @ x

        # add AWGN noise
        signal_power = Nt / Mr  # signal power per receive ant.: E(|xi|^2)=1 E(||hi||_F^2)=Nt
        sigma2 = signal_power * 10 ** (-(SNR) / 10)  # noise power per receive ant.; average SNR per receive ant.
        noise = np.sqrt(sigma2 / 2) * (np.random.randn(Mr, 1)
                                       + 1j * np.random.randn(Mr, 1))
        y = y + noise

        # convert complex into real
        x = np.concatenate((np.real(x), np.imag(x)))
        H = np.concatenate((np.concatenate((np.real(H), -np.imag(H)), axis=1),
                            np.concatenate((np.imag(H), np.real(H)), axis=1)))
        y = np.concatenate((np.real(y), np.imag(y)))

        sta = time.time()
        x_hat, MSE = detector(sysin, H, x, y, sigma2, MSE)
        end = time.time()

        # back into np.complex64
        x_hat = x_hat.reshape((2, Nt))
        x_hat = x_hat[0, :] + 1j * x_hat[1, :]

        # Demodulate
        x_hat_demod = QAM_Demodulation(x_hat, mu)

        total_time += (end - sta)

        # calculate BER
        err_bits = np.sum(np.not_equal(x_hat_demod, bits))
        total_err_bits += err_bits
        total_bits += mu * Nt
        count = count + 1
        if err_bits > 0:
            sys.stdout.write('\rtotal_err_bits={teb} total_bits={tb} BER={BER:.9f}'
                             .format(teb=total_err_bits, tb=total_bits, BER=total_err_bits / total_bits))
            sys.stdout.flush()
        if total_err_bits > err_bits_target or total_bits > 1e7:
            end = time.time()
            iter_time = end - start
            print("\nSNR=", SNR, "iter_time:", iter_time)
            ber = total_err_bits / total_bits
            print("BER:", ber)
            print("MSE:", 10 * np.log10(MSE / count))
            break

    # print("time:",total_time/1000)
    return ber, iter_time


def sample_gen(trainSet, ts, vs, training_flag=True):
    Mr, Nt = trainSet.Mr, trainSet.Nt
    mu, SNR = trainSet.mu, trainSet.snr
    CG = trainSet.CG
    channel_type, rho_tx, rho_rx = trainSet.channel_type, trainSet.rho_tx, trainSet.rho_rx
    if training_flag is False:
        ts = 0
    # generate training samples:
    H_ = np.zeros((2 * ts * Mr, 2 * Nt), dtype=np.float32)
    x_ = np.zeros((2 * ts * Nt, 1), dtype=np.float32)
    y_ = np.zeros((2 * ts * Mr, 1), dtype=np.float32)
    sigma2_ = np.zeros((ts, 1), dtype=np.float32)
    # generate development samples:
    Hval_ = np.zeros((2 * vs * Mr, 2 * Nt), dtype=np.float32)
    xval_ = np.zeros((2 * vs * Nt, 1), dtype=np.float32)
    yval_ = np.zeros((2 * vs * Mr, 1), dtype=np.float32)
    sigma2val_ = np.zeros((vs, 1), dtype=np.float32)

    if channel_type == 'corr':
        sqrtRtx, sqrtRrx = corr_channel(Mr, Nt, rho_tx=rho_tx, rho_rx=rho_rx)

    for i in range(ts + vs):
        # generate bits and modulate
        bits = np.random.binomial(n=1, p=0.5, size=(Nt * mu,))  # label
        bits_mod = QAM_Modulation(bits, mu)
        x = bits_mod.reshape(Nt, 1)
        # Rayleigh MIMO channel
        H = np.sqrt(1 / 2 / Mr) * (np.random.randn(Mr, Nt) +
                                   1j * np.random.randn(Mr, Nt))
        if channel_type == 'corr':  # correlated MIMO channel
            H = sqrtRrx @ H @ sqrtRtx
        # channel input & output
        y = H @ x
        signal_power = Nt / Mr  # signal power per receive ant.: E(|xi|^2)=1 E(||hi||_F^2)=Nt
        sigma2 = signal_power * 10 ** (-SNR / 10)  # noise power per receive ant.; average SNR per receive ant.
        noise = np.sqrt(sigma2 / 2) * (np.random.randn(Mr, 1)
                                       + 1j * np.random.randn(Mr, 1))
        y = y + noise

        # convert complex into real
        x = np.concatenate((np.real(x), np.imag(x)))
        H = np.concatenate((np.concatenate((np.real(H), -np.imag(H)), axis=1),
                            np.concatenate((np.imag(H), np.real(H)), axis=1)))
        y = np.concatenate((np.real(y), np.imag(y)))

        # stack
        if i < ts:
            H_[2 * Mr * i:2 * Mr * (i + 1)] = H
            x_[2 * Nt * i:2 * Nt * (i + 1)] = x
            y_[2 * Mr * i:2 * Mr * (i + 1)] = y
            sigma2_[i] = sigma2
        else:
            Hval_[2 * Mr * (i - ts):2 * Mr * (i - ts + 1)] = H
            xval_[2 * Nt * (i - ts):2 * Nt * (i - ts + 1)] = x
            yval_[2 * Mr * (i - ts):2 * Mr * (i - ts + 1)] = y
            sigma2val_[i - ts] = sigma2
    # reshape
    H_ = H_.reshape(ts, 2 * Mr, 2 * Nt).astype(np.float32)
    x_ = x_.reshape(ts, 2 * Nt, 1).astype(np.float32)
    y_ = y_.reshape(ts, 2 * Mr, 1).astype(np.float32)
    sigma2_ = sigma2_.reshape(ts, 1, 1).astype(np.float32)
    Hval_ = Hval_.reshape(vs, 2 * Mr, 2 * Nt).astype(np.float32)
    xval_ = xval_.reshape(vs, 2 * Nt, 1).astype(np.float32)
    yval_ = yval_.reshape(vs, 2 * Mr, 1).astype(np.float32)
    sigma2val_ = sigma2val_.reshape(vs, 1, 1).astype(np.float32)

    if CG is True:  # dataset for CG-OAMP-NET
        HHT = H_ @ np.transpose(H_, (0, 2, 1))
        C = HHT[:, :Mr, :Mr] + 1j * HHT[:, Mr:2 * Mr, :Mr]
        eig_ = np.linalg.eigvalsh(C).astype(np.float32).reshape((ts, Mr, 1))  # ts*Mr*1
        HHT = Hval_ @ np.transpose(Hval_, (0, 2, 1))
        C = HHT[:, :Mr, :Mr] + 1j * HHT[:, Mr:2 * Mr, :Mr]
        eigval_ = np.linalg.eigvalsh(C).astype(np.float32).reshape((vs, Mr, 1))  # vs*Mr*1
        return y_, x_, H_, sigma2_, eig_, yval_, xval_, Hval_, sigma2val_, eigval_
    else:  # dataset for OAMP-NET
        return y_, x_, H_, sigma2_, yval_, xval_, Hval_, sigma2val_


def corr_channel(Mr, Nt, rho_tx=0.5, rho_rx=0.5):
    Rtx_vec = np.ones(Nt)
    for i in range(1, Nt):
        Rtx_vec[i] = rho_tx ** i
    Rtx = toeplitz(Rtx_vec)
    if Mr == Nt and rho_tx == rho_rx:
        Rrx = Rtx
    else:
        Rrx_vec = np.ones(Mr)
        for i in range(1, Mr):
            Rrx_vec[i] = rho_rx ** i
        Rrx = toeplitz(Rrx_vec)

    # another way of constructing kronecker model
    # C = cholesky(np.kron(Rtx,Rrx))    # complex correlation
    # C = sqrtm(np.sqrt(np.kron(Rtx, Rrx)))  # power field correlation--what's an equivalent model?
    # return C

    sqrtRtx = sqrtm(Rtx)  # sqrt decomposition for power field

    if Mr == Nt and rho_tx == rho_rx:
        sqrtRrx = sqrtRtx
    else:
        sqrtRrx = sqrtm(Rrx)

    return sqrtRtx, sqrtRrx


def detector(sys, H, x, y, sigma2, MSE):
    detect_type = sys.detect_type
    if sys.use_OFDM:
        Mr, Nt = sys.Mr * sys.K, sys.Nt * sys.K
    else:
        Mr, Nt = sys.Mr, sys.Nt
    mu = sys.mu
    T = sys.T
    sess, prob, x_hat_net = sys.sess, sys.prob, sys.x_hat_net
    if detect_type is 'CG_OAMP_NET' or detect_type == 'OAMP_NET':  # use NET
        y = y.reshape((1, 2 * Mr, 1)).astype(np.float32)
        H_bar = H.reshape((1, 2 * Mr, 2 * Nt)).astype(np.float32)
        sigma2 = sigma2 * np.ones((1, 1, 1), dtype=np.float32)
        if detect_type is 'CG_OAMP_NET':  # CG_OAMP_NET
            HHT = H @ H.T
            # use complex value with half dimension for faster eigendecomposition
            C = (HHT[:Mr, :Mr] + 1j * HHT[Mr:2 * Mr, :Mr])  # .reshape((1,Mr,Mr)).astype(np.complex64)
            eigval = np.linalg.eigvalsh(C).astype(np.float32).reshape((1, Mr, 1))
            x_hat = sess.run(x_hat_net, feed_dict={prob.y_: y,
                                                   prob.x_: np.zeros((1, 2 * Nt, 1), dtype=np.float32), prob.H_: H_bar,
                                                   prob.sigma2_: sigma2, prob.sample_size_: 1,
                                                   prob.eigvalue_: eigval})
            # prob.sigma2_: sigma2, prob.sample_size_: 1, prob.C_: C})
        else:  # OAMP-NET
            x_hat = sess.run(x_hat_net, feed_dict={prob.y_: y,
                                                   prob.x_: np.zeros((1, 2 * Nt, 1), dtype=np.float32), prob.H_: H_bar,
                                                   prob.sigma2_: sigma2, prob.sample_size_: 1})
    elif detect_type == 'CG_OAMP':
        x_hat, _, mse = CG_OAMP(x, H, y, sigma2 / 2, I=50, T=T, mu=mu)
        MSE += mse
    elif detect_type == 'OAMP':
        x_hat, mse = OAMP(x, H, y, sigma2 / 2, T=T, mu=mu)
        MSE += mse
    elif detect_type == 'MAMP':
        s = la.svd(H, compute_uv=False)
        lambda_dag = (max(s) ** 2 + min(s) ** 2) / 2
        x_hat, mse = MAMP(x, H, s, y, lambda_dag, sigma2 / 2, L=3, T=T, mu=mu)
        MSE += mse
    elif detect_type == 'ZF':  # ZF
        HT = H.T
        x_hat = la.inv(HT @ H) @ HT @ y
    elif detect_type == 'MMSE':  # MMSE
        HT = H.T
        x_hat = la.inv(HT @ H + sigma2 / 2 * np.eye(2 * Nt)) @ HT @ y
    else:
        raise RuntimeError('The selected detector does not exist!')

    return x_hat, MSE
