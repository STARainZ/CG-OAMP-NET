#!/usr/bin/python
from __future__ import division
from __future__ import print_function
from tools import problems, networks

import numpy as np
import scipy.io as sio
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

np.random.seed(1)  # numpy is good about making repeatable output
tf.set_random_seed(1)

# system parameters

SNR_train = [0, 5, 10, 15, 20, 25, 30]  # SNR list
test = True  # test or train
BER = []


class SysIn(object):
    mu = 2  # modulation order:2^mu QPSK:mu=2 16QAM:mu=4
    Mr = 8  # number of receiving antennas
    Nt = 8  # number of transmitting antennas
    use_OFDM = False  # whether to use OFDM
    K = 64  # number of subcarriers
    CP = 16  # CP length
    CP_flag = True  # with or without CP; note: when CP_flag is False, CP should be set to zero
    T, icg = 4, 50  # number of iterations
    detect_type = 'CG_OAMP'  # 'CG_OAMP_NET', 'OAMP_NET', 'CG_OAMP', 'OAMP', 'MAMP', 'ZF' or 'MMSE'
    channel_type = 'rayleigh'  # channel type: 'rayleigh' (default), 'corr' or 'winner'
    rho_tx, rho_rx = 0.5, 0.5  # correlation coefficients for spatial correlated channels
    filename = 'BER_8MIMO_QPSK_' + detect_type  # for saving the simulation results
    problem_size = [Mr * K, Nt * K] if use_OFDM else [Mr, Nt]
    sess, prob, x_hat_net = [], problems.MIMO_detection_problem(problem_size[0], problem_size[1]), []


# Note: for the test of the network, the training settings should be the same as the system setup
class TrainSetting(object):
    CG = True  # CG-OAMP-NET or OAMP-NET
    T, icg = 4, 50  # number of iterations
    version = 1  # four trainable parameters
    lr = 1e-3  # initial learning rate
    maxit = 1000  # number of epoches
    sample_size, vsample_size = 5000, 10000  # training and validation sample size in an epoch
    total_batch = 5
    batch_size = int(sample_size / total_batch)
    snr = 20  # training SNR
    channel_type = 'rayleigh'  # channel type: 'rayleigh', 'corr' or 'winner'
    rho_tx, rho_rx = 0.5, 0.5
    savefile = 'CG_OAMP_QPSK_8_8_20dB_T4.npz'


""" The training setup for reference """
# PC, NVIDIA GeForce GTX 1050 Ti 8GB, 16 GB RAM
# 8MIMO: samplesize=500  batch_size=5 vsamplesize=1000
# 32MIMO:samplesize=150 batch_size=25

# server, NVIDIA GeForce GTX 1080 Ti 10GB
# 8MIMO: samplesize=5000  batch_size=1000 vsamplesize=20000
# 32MIMO:samplesize=5000  batch_size=1000 vsamplesize=10000
# 128MIMO:samplesize=1000  batch_size=200 vsamplesize=3000
# 8MIMO-OFDM:samplesize=80*64 batch_size=2*64 vsamplesize=320*64
# CP-free OTA: 20 \times 16:samplesize=50 batch_size=10 vsamplesize=10


sys = SysIn()
trainSet = TrainSetting()
trainSet.mu, trainSet.Mr, trainSet.Nt, trainSet.use_OFDM, trainSet.K, trainSet.CP, trainSet.CP_flag =\
    sys.mu, sys.Mr, sys.Nt, sys.use_OFDM, sys.K, sys.CP, sys.CP_flag
trainSet.prob = sys.prob
trainSet.test = test


# train
if test is False:
    if trainSet.CG:
        """ CG-OAMP-NET """
        sys.sess, sys.x_hat_net = networks.build_CG_OAMP(trainSet)
    else:
        """ OAMP-NET """
        sys.sess, sys.x_hat_net = networks.build_OAMP(trainSet)

# test
else:
    if sys.detect_type is 'CG_OAMP_NET':  # load the trained model
        sys.sess, sys.x_hat_net = networks.build_CG_OAMP(trainSet)
    elif sys.detect_type is 'OAMP_NET':  # load the trained model
        sys.sess, sys.x_hat_net = networks.build_OAMP(trainSet)
    if sys.use_OFDM is True:
        from tools import MIMO_OFDM_detection

        print('Full CP' if sys.CP_flag else 'CP-free')
    else:
        from tools import MIMO_detection
    for i in range(7):
        print("SNR=", SNR_train[i])
        np.random.seed(1)
        if sys.use_OFDM is False:  # single-carrier MIMO detection
            ber, _ = MIMO_detection.MIMO_detection_simulate(sys, SNR=SNR_train[i])
        else:  # MIMO-OFDM
            ber, _ = MIMO_OFDM_detection.MIMO_OFDM_simulate(sys, SNR=SNR_train[i])
        BER.append(ber)

print(BER)
BER_matlab = np.array(BER)

# save the BER results -- example
# sio.savemat(sys.filename+'.mat',{sys.filename: BER_matlab})
