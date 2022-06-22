#!/usr/bin/python
from __future__ import division
import numpy as np
import time
import sys
import numpy.linalg as la
from scipy.linalg import dft
from .utils import QAM_Modulation, QAM_Demodulation, addCP, Clipping
from .MIMO_detection import detector

""" WINNER II """
channel_train = np.load('winner_model/channel_train.npy')
train_size = channel_train.shape[0]  # 100000
channel_test = np.load('winner_model/channel_test.npy')
test_size = channel_test.shape[0]  # 390000
L = channel_train.shape[1]  # channel delay spread


def ofdm_simulate_symbol_by_symbol(codeword, SNRdb, mu, CP_flag, K, CP,
                                   pilots=False, Clipping_Flag=False, CR=1):
    if pilots:
        codeword_qam = codeword
    else:
        codeword_qam = QAM_Modulation(codeword, mu)
    if len(codeword_qam) != K:
        print('length of code word is not equal to K, error !!')
    OFDM_time_codeword = np.sqrt(K) * np.fft.ifft(codeword_qam)
    if CP_flag is False:
        OFDM_withCP_codeword = OFDM_time_codeword
    else:
        OFDM_withCP_codeword = addCP(OFDM_time_codeword, CP, CP_flag, mu, K)
    if Clipping_Flag:
        OFDM_withCP_codeword = Clipping(OFDM_withCP_codeword, CR)  # add clipping

    return OFDM_withCP_codeword, codeword_qam


def get_MIMO_rx_signal(h_total, SNR, Mr=8, Nt=8, mu=2, K=64, CP=16,
                       L_conv=None, bits=None, pilots=False, training=False,
                       CP_flag=True, ISI_symbol=None, L=None):
    # global ISI_symbol
    rx_sum_withCP = np.zeros((L_conv, Mr), dtype=complex)
    rx_withCP = np.zeros((Nt * L_conv, Mr), dtype=complex)
    bits_mod = np.zeros((Nt, K), dtype=complex)

    for i in range(Nt):
        OFDM_withCP_codeword, bits_mod[i] = ofdm_simulate_symbol_by_symbol(bits[i], SNR, mu,
                                                                           CP_flag, K, CP, pilots=pilots)
        for j in range(Mr):
            convolved = np.convolve(OFDM_withCP_codeword, h_total[i * L:(i + 1) * L, j].T)
            rx_withCP[i * L_conv:(i + 1) * L_conv, j] = convolved  # rx without noise
    # receiving signal is the sum of transmitted signal from different antennas
    for i in range(Nt):
        rx_sum_withCP += rx_withCP[i * L_conv:(i + 1) * L_conv]

    rx_withCP = rx_sum_withCP
    rx_withCP_raw = rx_withCP

    signal_power = np.mean(abs(rx_withCP ** 2), axis=0)
    sigma2 = signal_power * 10 ** (-SNR / 10)  # 1*Mr [sigma_1,...,sigma_Mr]
    noise = np.sqrt(sigma2 / 2).reshape(1, rx_withCP.shape[1]) * \
            (np.random.randn(*rx_withCP.shape) + 1j * np.random.randn(*rx_withCP.shape))
    rx_withCP += noise
    sigma2 = np.mean(sigma2)

    if CP_flag is False and training is False:
        # add ISI
        rx_withCP += ISI_symbol
        # update ISI
        ISI_symbol[0:(L - 1)] = rx_withCP[-(L - 1):]  # ISI includes noise from the previous symbol

    # remove CP
    rx = rx_withCP[CP:(CP + K)]  # rx:K*M

    return rx, sigma2, rx_withCP_raw, bits_mod, ISI_symbol


def get_circ_and_cutoff_channel_mat(sys, h_total, FH_kron):
    Mr, Nt, K = sys.Mr, sys.Nt, sys.K
    CP, CP_flag, L = sys.CP, sys.CP_flag, sys.L
    J = np.zeros((Mr * K, Nt * K), dtype=complex)
    h = np.zeros((L, Mr, Nt), dtype=complex)
    for l in range(L):
        for m in range(Mr):
            for n in range(Nt):
                h[l, m, n] = h_total[n * L + l, m]
    # get MIMO OFDM cp-free channel matrix J = H-A
    for j in range(K):
        for i in range(L):
            if j < K - L + 1 or (i + j) < K:
                J[(i + j) * Mr:(i + j + 1) * Mr, j * Nt:(j + 1) * Nt] = h[i]

    # signal processing in time domain first
    A = np.zeros((Mr * K, Nt * K), dtype=complex)
    for j in range(L - 1):
        for i in range(L - 1):
            if i + j + 1 <= L - 1:
                A[i * Mr:(i + 1) * Mr, (K - j - 1) * Nt:(K - j) * Nt] = h[i + j + 1]

    if (L > CP + 1 or CP_flag is False):
        H = J @ FH_kron
    else:
        H = (J + A) @ FH_kron
    H_bar = np.concatenate((np.concatenate((np.real(H), -np.imag(H)), axis=1),
                            np.concatenate((np.imag(H), np.real(H)), axis=1)))
    return H_bar, A


def MIMO_OFDM_simulate(sysin, SNR=40):
    """
    We assume that the channel does not change during a slot (7 MIMO-OFDM symbols), and ISI only exists in a frame.
    Args:
        sysin ():
        SNR ():

    Returns:

    """
    Mr, Nt, mu = sysin.Mr, sysin.Nt, sysin.mu
    K, CP, CP_flag = sysin.K, sysin.CP, sysin.CP_flag
    sysin.L = L
    err_bits_target = 1000
    total_err_bits = 0
    total_bits = 0
    count = 0
    MSE = 0
    start = time.time()

    num_of_symbol = 7  # number of symbols for one slot
    L_conv = K + CP + L - 1  # symbol length after convolution
    Buffer = np.zeros((Nt, K), dtype=complex)  # buffering the previous estimated OFDM symbol to remove ISI

    """ convolutional code and viterbi decode """
    payloadBits_per_OFDM = K * mu

    F = dft(K) / np.sqrt(K)
    FH = np.conj(F.T)
    FH_kron = np.kron(FH, np.eye(Nt))

    total_time = 0.

    while True:
        if count % num_of_symbol == 0:
            ISI_symbol = np.zeros((L_conv, Mr), dtype=complex)
            if CP_flag is False:
                Buffer = np.zeros((Nt, K), dtype=complex)

            index = np.random.choice(np.arange(test_size), size=Mr * Nt)  # select Mr*Nt index from the test set
            h_total = channel_test[index]  # MN*L
            h_total = h_total.reshape((Mr, Nt * L)).T  # h10 = h[1][0*L:1*L]

            H_bar, A = get_circ_and_cutoff_channel_mat(sysin, h_total, FH_kron)

        bits = np.random.binomial(n=1, p=0.5, size=(Nt, payloadBits_per_OFDM))  # label
        x_demod = np.zeros((Nt, payloadBits_per_OFDM), dtype=int)
        y, sigma2, _, bits_mod, ISI_symbol = get_MIMO_rx_signal(h_total, SNR, Mr=Mr, Nt=Nt, mu=mu,
                                                                K=K, CP=CP, L_conv=L_conv, bits=bits,
                                                                CP_flag=CP_flag, ISI_symbol=ISI_symbol, L=L)
        sigma2 = np.mean(sigma2)
        x = bits_mod.reshape((Nt * K, 1), order='F')
        x = np.concatenate((np.real(x), np.imag(x)))
        y = y.reshape(Mr * K, 1)

        if count % num_of_symbol == 0:
            pass
        else:
            if (L > CP + 1 or CP_flag is False):
                y -= A @ FH_kron @ Buffer.reshape((Nt * K, 1), order='F')  # Remove ISI

        y = np.concatenate((np.real(y), np.imag(y)))

        sta = time.time()
        """detector"""
        x_hat, MSE = detector(sysin, H_bar, x, y, sigma2, MSE)
        end = time.time()
        total_time += end - sta

        """hard decision"""
        x_hat = x_hat.reshape(2, Nt * K)
        x_hat = x_hat[0, :] + 1j * x_hat[1, :]
        x_hat_demod = QAM_Demodulation(x_hat, mu)
        for k in range(K):
            x_demod[:, k * mu:(k + 1) * mu] = x_hat_demod[k * Nt * mu:(k + 1) * Nt * mu].reshape(Nt, mu)

        # buffering current detected symbol in frequency domain
        for i in range(Nt):
            Buffer[i] = QAM_Modulation(x_demod[i], mu)

        # calculate BER
        err_bits = np.sum(np.not_equal(x_demod, bits))
        total_err_bits += err_bits
        total_bits += Nt * payloadBits_per_OFDM
        count = count + 1
        if err_bits > 0:
            sys.stdout.write('\rtotal_err_bits={teb} total_bits={tb} BER={BER:.9f}'
                             .format(teb=total_err_bits, tb=total_bits,
                                     BER=total_err_bits / total_bits))
            sys.stdout.flush()
        if total_err_bits > err_bits_target or total_bits > 1e7:
            end = time.time()
            iter_time = end - start
            print("\nSNR=", SNR, "iter_time:", iter_time)
            ber = total_err_bits / total_bits
            print("BER:", ber)
            print("MSE:", 10 * np.log10(MSE / count))
            break
    # print("time:", total_time / num_of_trails)
    return ber, iter_time


def sample_gen_MIMO_OFDM(trainSet, ss, training_flag=True):
    Mr, Nt = trainSet.Mr, trainSet.Nt
    mu, SNR = trainSet.mu, trainSet.snr
    K, CP, CP_flag = trainSet.K, trainSet.CP, trainSet.CP_flag
    CG = trainSet.CG
    trainSet.L = L
    if training_flag:
        # generate training channels:
        index = np.random.choice(np.arange(train_size), size=ss * Mr * Nt)
        h_total = np.transpose(channel_train[index].reshape(ss, Mr, Nt * L),
                               (0, 2, 1)).astype(np.complex64)
    else:
        # generate development channels:
        index = np.random.choice(np.arange(test_size), size=ss * Mr * Nt)
        h_total = np.transpose(channel_test[index].reshape(ss, Mr, Nt * L),
                               (0, 2, 1)).astype(np.complex64)
    L_conv = K + CP + L - 1
    payloadBits_per_OFDM = K * mu
    count = 0

    H_ = np.zeros((ss, 2 * Mr * K, 2 * Nt * K), dtype=np.float32)
    x_ = np.zeros((ss, 2 * Nt * K, 1), dtype=np.float32)
    y_ = np.zeros((ss, 2 * Mr * K, 1), dtype=np.float32)
    sigma2_ = np.zeros((ss, 1, 1), dtype=np.float32)
    FH_kron = np.kron(np.conj(dft(K).T) / np.sqrt(K), np.eye(Nt))

    for hs in h_total:
        bits = np.random.binomial(n=1, p=0.5, size=(Nt, payloadBits_per_OFDM))  # label
        y, sigma2, _, bits_mod, _ = get_MIMO_rx_signal(hs, SNR, Mr=Mr, Nt=Nt, mu=mu,
                                                       K=K, CP=CP, L_conv=L_conv, bits=bits, CP_flag=CP_flag, L=L,
                                                       training=True)
        H_bar, _ = get_circ_and_cutoff_channel_mat(trainSet, hs, FH_kron)

        bits_mod = bits_mod.reshape((Nt * K, 1), order='F')
        x = np.concatenate((np.real(bits_mod), np.imag(bits_mod)))
        y = y.reshape((Mr * K, 1))
        yd = np.concatenate((np.real(y), np.imag(y)))
        sigma2 = np.mean(sigma2)

        # stack
        H_[count] = H_bar.astype(np.float32)
        x_[count] = x.astype(np.float32)
        y_[count] = yd.astype(np.float32)
        sigma2_[count] = sigma2.astype(np.float32)

        count += 1

    if CG is True:
        HHT = H_ @ np.transpose(H_, (0, 2, 1))
        if CP_flag is None:
            C = HHT[:, :Mr, :Mr] + 1j * HHT[:, Mr:2 * Mr, :Mr]
            eig_ = np.linalg.eigvalsh(C).astype(np.float32).reshape(ss * K, Mr, 1)  # ss*Mr*1
        else:
            C = HHT[:, :Mr * K, :Mr * K] + 1j * HHT[:, Mr * K:2 * Mr * K, :Mr * K]
            eig_ = np.linalg.eigvalsh(C).astype(np.float32).reshape(ss, Mr * K, 1)
        return y_, x_, H_, sigma2_, eig_
    else:
        return y_, x_, H_, sigma2_
