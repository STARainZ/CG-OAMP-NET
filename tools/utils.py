#!/usr/bin/python
from __future__ import division
import numpy as np
import math

sqrt = np.sqrt
pi = math.pi

_QPSK_mapping_table = {
    (0,1): (-1+1j,), (1,1): (1+1j,),
    (0,0): (-1-1j,), (1,0): (1-1j,)
}

_QPSK_demapping_table = {v: k for k, v in _QPSK_mapping_table.items()}

_QPSK_Constellation = np.array([[-1+1j], [1+1j],
                                [-1-1j], [1-1j]])

_16QAM_mapping_table = {
    (0,0,1,0): (-3+3j,), (0,1,1,0): (-1+3j,), (1,1,1,0): (1+3j,), (1,0,1,0): (3+3j,),
    (0,0,1,1): (-3+1j,), (0,1,1,1): (-1+1j,), (1,1,1,1): (1+1j,), (1,0,1,1): (3+1j,),
    (0,0,0,1): (-3-1j,), (0,1,0,1): (-1-1j,), (1,1,0,1): (1-1j,), (1,0,0,1): (3-1j,),
    (0,0,0,0): (-3-3j,), (0,1,0,0): (-1-3j,), (1,1,0,0): (1-3j,), (1,0,0,0): (3-3j,)
}

_16QAM_demapping_table = {v: k for k, v in _16QAM_mapping_table.items()}

_16QAM_Constellation = np.array([[-3+3j], [-1+3j], [1+3j], [3+3j],
                                 [-3+1j], [-1+1j], [1+1j], [3+1j],
                                 [-3-1j], [-1-1j], [1-1j], [3-1j],
                                 [-3-3j], [-1-3j], [1-3j], [3-3j]])

_64QAM_mapping_table = {
    (0,0,0,1,0,0): (-7+7j,), (0,0,1,1,0,0): (-5+7j,), (0,1,1,1,0,0): (-3+7j,), (0,1,0,1,0,0): (-1+7j,), (1,1,0,1,0,0): (1+7j,), (1,1,1,1,0,0): (3+7j,), (1,0,1,1,0,0): (5+7j,), (1,0,0,1,0,0): (7+7j,),
    (0,0,0,1,0,1): (-7+5j,), (0,0,1,1,0,1): (-5+5j,), (0,1,1,1,0,1): (-3+5j,), (0,1,0,1,0,1): (-1+5j,), (1,1,0,1,0,1): (1+5j,), (1,1,1,1,0,1): (3+5j,), (1,0,1,1,0,1): (5+5j,), (1,0,0,1,0,1): (7+5j,),
    (0,0,0,1,1,1): (-7+3j,), (0,0,1,1,1,1): (-5+3j,), (0,1,1,1,1,1): (-3+3j,), (0,1,0,1,1,1): (-1+3j,), (1,1,0,1,1,1): (1+3j,), (1,1,1,1,1,1): (3+3j,), (1,0,1,1,1,1): (5+3j,), (1,0,0,1,1,1): (7+3j,),
    (0,0,0,1,1,0): (-7+1j,), (0,0,1,1,1,0): (-5+1j,), (0,1,1,1,1,0): (-3+1j,), (0,1,0,1,1,0): (-1+1j,), (1,1,0,1,1,0): (1+1j,), (1,1,1,1,1,0): (3+1j,), (1,0,1,1,1,0): (5+1j,), (1,0,0,1,1,0): (7+1j,),
    (0,0,0,0,1,0): (-7-1j,), (0,0,1,0,1,0): (-5-1j,), (0,1,1,0,1,0): (-3-1j,), (0,1,0,0,1,0): (-1-1j,), (1,1,0,0,1,0): (1-1j,), (1,1,1,0,1,0): (3-1j,), (1,0,1,0,1,0): (5-1j,), (1,0,0,0,1,0): (7-1j,),
    (0,0,0,0,1,1): (-7-3j,), (0,0,1,0,1,1): (-5-3j,), (0,1,1,0,1,1): (-3-3j,), (0,1,0,0,1,1): (-1-3j,), (1,1,0,0,1,1): (1-3j,), (1,1,1,0,1,1): (3-3j,), (1,0,1,0,1,1): (5-3j,), (1,0,0,0,1,1): (7-3j,),
    (0,0,0,0,0,1): (-7-5j,), (0,0,1,0,0,1): (-5-5j,), (0,1,1,0,0,1): (-3-5j,), (0,1,0,0,0,1): (-1-5j,), (1,1,0,0,0,1): (1-5j,), (1,1,1,0,0,1): (3-5j,), (1,0,1,0,0,1): (5-5j,), (1,0,0,0,0,1): (7-5j,),
    (0,0,0,0,0,0): (-7-7j,), (0,0,1,0,0,0): (-5-7j,), (0,1,1,0,0,0): (-3-7j,), (0,1,0,0,0,0): (-1-7j,), (1,1,0,0,0,0): (1-7j,), (1,1,1,0,0,0): (3-7j,), (1,0,1,0,0,0): (5-7j,), (1,0,0,0,0,0): (7-7j,)
}

_64QAM_demapping_table = {v: k for k, v in _64QAM_mapping_table.items()}

_64QAM_Constellation = np.array([[-7+7j], [-5+7j], [-3+7j], [-1+7j], [1+7j], [3+7j], [5+7j], [7+7j],
                                [-7+5j], [-5+5j], [-3+5j], [-1+5j], [1+5j], [3+5j], [5+5j], [7+5j],
                                [-7+3j], [-5+3j], [-3+3j], [-1+3j], [1+3j], [3+3j], [5+3j], [7+3j],
                                [-7+1j], [-5+1j], [-3+1j], [-1+1j], [1+1j], [3+1j], [5+1j], [7+1j],
                                [-7-1j], [-5-1j], [-3-1j], [-1-1j], [1-1j], [3-1j], [5-1j], [7-1j],
                                [-7-3j], [-5-3j], [-3-3j], [-1-3j], [1-3j], [3-3j], [5-3j], [7-3j],
                                [-7-5j], [-5-5j], [-3-5j], [-1-5j], [1-5j], [3-5j], [5-5j], [7-5j],
                                [-7-7j], [-5-7j], [-3-7j], [-1-7j], [1-7j], [3-7j], [5-7j], [7-7j]])

sq2 = sqrt(2)
sq10 = sqrt(10)
sq42 = sqrt(42)


def Clipping(x, CL):
    sigma = np.sqrt(np.mean(np.square(np.abs(x))))  # RMS of OFDM signal
    CL = CL*sigma   # clipping level
    x_clipped = x
    clipped_idx = abs(x_clipped) > CL
    x_clipped[clipped_idx] = np.divide((x_clipped[clipped_idx]*CL),abs(x_clipped[clipped_idx]))
    #print (sum(abs(x_clipped_temp-x_clipped)))
    return x_clipped


def PAPR(x):
    Power = np.abs(x)**2
    PeakP = np.max(Power)
    AvgP = np.mean(Power)
    PAPR_dB = 10*np.log10(PeakP/AvgP)
    return PAPR_dB


def QAM_Modulation(bits,mu):
    if mu == 1:
        bits_mod = (2*bits-1).reshape(int(len(bits)),1)
    elif mu == 2:
        bits_mod = Modulation(bits)/sq2
    elif mu == 4:
        bits_mod = Modulation_16(bits)/sq10
    else:
        bits_mod = Modulation_64(bits)/sq42
    return bits_mod


def Modulation(bits):
    bit_r = bits.reshape((int(len(bits)/2), 2))  # real & imag
    return (2*bit_r[:,0]-1)+1j*(2*bit_r[:,1]-1)  # This is just for QAM modulation
#    return np.concatenate((2*bit_r[:,0]-1, 2*bit_r[:,1]-1))


# mapping
def Modulation_16(bits):
    bit_r = bits.reshape((int(len(bits)/4), 4))
    bit_mod = []
    for i in range(int(len(bits)/4)):
        bit_mod.append(list(_16QAM_mapping_table.get(tuple(bit_r[i]))))
    return np.asarray(bit_mod).reshape((-1,))


def Modulation_64(bits):
    bit_r = bits.reshape((int(len(bits)/6), 6))
    bit_mod = []
    for i in range(int(len(bits)/6)):
        bit_mod.append(list(_64QAM_mapping_table.get(tuple(bit_r[i]))))
    return np.asarray(bit_mod).reshape((-1,))


def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)


def addCP(OFDM_time, CP, CP_flag, mu, K):
    if CP == 0:
        return OFDM_time
    elif CP_flag is False:
        # add noise CP——no ISI, only ICI
        bits_noise = np.random.binomial(n=1, p=0.5, size=(K*mu, ))
        codeword_noise = QAM_Modulation(bits_noise,mu)
        OFDM_time_noise = np.fft.ifft(codeword_noise)
        cp = OFDM_time_noise[-CP:]
    else:
        cp = OFDM_time[-CP:]               # take the last CP samples ...
    return np.hstack([cp, OFDM_time])  # ... and add them to the beginning


def channel(signal,channelResponse,SNRdb):
    convolved = np.convolve(signal, channelResponse)
    signal_power = np.mean(abs(convolved**2))
    sigma2 = signal_power * 10**(-SNRdb/10)
    noise = np.sqrt(sigma2/2) * (np.random.randn(*convolved.shape) +
                                 1j*np.random.randn(*convolved.shape))
    return convolved + noise,sigma2


def removeCP(signal, CP, K):
    return signal[CP:(CP+K)]    # cp~cp+K


def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX)


def equalize(OFDM_demod, Hest):
    return OFDM_demod / Hest


def QAM_Demodulation(bits_mod,mu):
    if mu == 1:
        bits_demod = abs(bits_mod+1) >= abs(bits_mod-1)
        bits_demod = bits_demod.astype(np.int32).reshape(-1)
    elif mu == 2:
        bits_demod = Demodulation(bits_mod*sq2)
    elif mu == 4:
        bits_demod = Demodulation_16(bits_mod*sq10)
    else:
        bits_demod = Demodulation_64(bits_mod*sq42)
    return bits_demod


def Demodulation(bits_mod):
    X_pred = np.array([])
    for i in range(len(bits_mod)):
        tmp = bits_mod[i] * np.ones((4,1))
        min_distance_index = np.argmin(abs(tmp - _QPSK_Constellation))
        X_pred = np.concatenate((X_pred,np.array(_QPSK_demapping_table[
            tuple(_QPSK_Constellation[min_distance_index])])))
    return X_pred


def Demodulation_16(bits_mod):
    X_pred = np.array([])
    for i in range(len(bits_mod)):
        tmp = bits_mod[i] * np.ones((16,1))
        min_distance_index = np.argmin(abs(tmp - _16QAM_Constellation))
        X_pred = np.concatenate((X_pred,np.array(_16QAM_demapping_table[
            tuple(_16QAM_Constellation[min_distance_index])])))
    return X_pred


def Demodulation_64(bits_mod):
    X_pred = np.array([])
    for i in range(len(bits_mod)):
        tmp = bits_mod[i] * np.ones((64,1))
        min_distance_index = np.argmin(abs(tmp - _64QAM_Constellation))
        X_pred = np.concatenate((X_pred,np.array(_64QAM_demapping_table[
            tuple(_64QAM_Constellation[min_distance_index])])))
    return X_pred


def PS(bits):
    return bits.reshape((-1,))


def NLE(vle,ule,orth=True,mu=2,SE=False,x=None,EP=False):
    # for QPSK signal
    if mu == 2:  # {-1,+1}
        P0 = np.maximum(np.exp(-(-1/sq2-ule)**2/(2*vle)),1e-100)
        P1 = np.maximum(np.exp(-(1/sq2-ule)**2/(2*vle)),1e-100)
        u_post = (P1-P0) / (P1+P0)/sq2
        if SE is True:
            v_post = np.mean((x-u_post)**2)
        else:
            v_post = (P0*(u_post+1/sq2)**2+P1*(u_post-1/sq2)**2)/(P1+P0)
    elif mu == 4:  # {-3,-1,+1,+3}
        P_3 = np.maximum(np.exp(-(-3/sq10-ule)**2/(2*vle)),1e-100)
        P_1 = np.maximum(np.exp(-(-1/sq10-ule)**2/(2*vle)),1e-100)
        P1 = np.maximum(np.exp(-(1/sq10-ule)**2/(2*vle)),1e-100)
        P3 = np.maximum(np.exp(-(3/sq10-ule)**2/(2*vle)),1e-100)
        u_post = (-3*P_3-P_1+P1+3*P3) / (P_3+P_1+P1+P3)/sq10
        if SE is True:
            v_post = np.mean((x-u_post)**2)
        else:
            v_post = (P_3*(u_post+3/sq10)**2+P_1*(u_post+1/sq10)**2 +
                      P1*(u_post-1/sq10)**2+P3*(u_post-3/sq10)**2)/(P_3+P_1+P1+P3)
    else:  # {-7,-5,-3,-1,+1,+3,+5,+7}
        P_7 = np.maximum(np.exp(-(-7/sq42-ule)**2/(2*vle)),1e-100)
        P_5 = np.maximum(np.exp(-(-5/sq42-ule)**2/(2*vle)),1e-100)
        P_3 = np.maximum(np.exp(-(-3/sq42-ule)**2/(2*vle)),1e-100)
        P_1 = np.maximum(np.exp(-(-1/sq42-ule)**2/(2*vle)),1e-100)
        P1 = np.maximum(np.exp(-(1/sq42-ule)**2/(2*vle)),1e-100)
        P3 = np.maximum(np.exp(-(3/sq42-ule)**2/(2*vle)),1e-100)
        P5 = np.maximum(np.exp(-(5/sq42-ule)**2/(2*vle)),1e-100)
        P7 = np.maximum(np.exp(-(7/sq42-ule)**2/(2*vle)),1e-100)
        u_post = (-7*P_7-5*P_5-3*P_3-P_1+P1+3*P3+5*P5+7*P7) / \
                 (P_7+P_5+P_3+P_1+P1+P3+P5+P7)/sq42
        if SE is True:
            v_post = np.mean((x-u_post)**2)
        else:
            v_post = (P_7*(u_post+7/sq42)**2+P_5*(u_post+5/sq42)**2 +
                      P_3*(u_post+3/sq42)**2+P_1*(u_post+1/sq42)**2 +
                      P1*(u_post-1/sq42)**2+P3*(u_post-3/sq42)**2 +
                      P5*(u_post-5/sq42)**2+P7*(u_post-7/sq42)**2) / \
                     (P_7+P_5+P_3+P_1+P1+P3+P5+P7)
    if EP is False:
        v_post = np.mean(v_post)

    if orth:
        u_orth = (u_post/v_post-ule/vle)/(1/v_post-1/vle)
        v_orth = 1/(1/v_post-1/vle)
    else:
        u_orth = u_post
        v_orth = v_post

    return u_post,v_post,u_orth,v_orth

