#!/usr/bin/python

import numpy as np
import numpy.linalg as la
from .utils import NLE


def MAMP(x, A, s, y, lambda_dag, noise_var, L=3, T=50, mu=2, orth=False):
    ''' initialization '''
    M = A.shape[0]
    N = A.shape[1]
    delta = M / N  # compressed ratio = M/N
    AH = A.T
    AAH = A @ AH
    B = lambda_dag - np.append(s ** 2, np.zeros(M - N if M >= N else 0))
    sign = np.zeros(M)
    sign[B > 0] = 1
    sign[B < 0] = -1
    log_B = np.log(abs(B))
    w0 = 1 / N * (lambda_dag * M - sum(B))
    w1 = 1 / N * (lambda_dag * sum(B) - sum(B ** 2))
    wbar00 = lambda_dag * w0 - w1 - w0 * w0
    zt = np.zeros((M, T), dtype=np.float64)  # z0~z(T-1)
    zt[:, 0] = y.T
    rhat_t = np.zeros_like(y, dtype=np.float64)
    xt = np.zeros((N, T), dtype=np.float64)  # x0~x(T-1)
    log_theta = np.zeros(T)  # 0<=t<=T-1,0<=i<=T-1
    log_theta_p = np.zeros(T)
    theta_w_ = np.zeros(2 * T - 1)
    p_ti = np.zeros(T, dtype=np.float64)  # 0<=t<=T-1,0<=i<=T-1
    xi_t = 1

    # calculate vphi_00
    vphi_tt = np.zeros((T, T), dtype=np.float64)  # 0<=t<=T-1
    vphi_tt[0, 0] = max((zt[:, 0].T @ zt[:, 0] / N - delta * noise_var) / w0, 1e-10)
    # for correlated
    # vphi_tt[0,0] = (zt[:,0].T@zt[:,0]/N - delta*noise_var)/w0

    damping_flag = True
    singular = False
    MSE = np.zeros(T, dtype=np.float64)
    VAR = np.zeros(T, dtype=np.float64)
    ''' iteration '''
    for t in range(T):  # 0<=t<=T-1
        ''' MLE '''
        theta_t = 1 / (lambda_dag + noise_var / vphi_tt[t, t])  # (33)
        # if theta_t<0:
        #     print(t)
        # calculate p_ti & xi_t (19ab)&(36)&(37)
        if t >= 1:
            log_theta_p[0:t] = log_theta[0:t]  # attention:have the same address
            log_theta[0:t] = log_theta_p[0:t] + np.log(theta_t)
            theta_w_[t - 1] = theta_w(lambda_dag, B, sign, log_B, log_theta[0], t, N)
            theta_w_[t] = theta_w(lambda_dag, B, sign, log_B, log_theta[0], t + 1, N)
            theta_w_[2 * t] = theta_w(lambda_dag, B, sign, log_B, 2 * log_theta[0],
                                      2 * t + 1, N)
            if t >= 2:
                theta_w_[0:t - 1] = theta_w_[0:t - 1] * np.exp(np.flipud(
                    log_theta[1:t] - log_theta_p[0:t - 1]))
                theta_w_[2 * t - 1] = theta_w(lambda_dag, B, sign, log_B, log_theta[0] +
                                              log_theta[1], 2 * t, N)
                if t >= 3:
                    theta_w_[t + 1:2 * t - 1] = theta_w_[t + 1:2 * t - 1] * np.exp(np.log(theta_t) +
                                                                                   np.flipud(log_theta[2:t]
                                                                                             - log_theta_p[0:t - 2]))
            p_ti[0:t] = np.flipud(theta_w_[0:t])
            c_t = Get_c(p_ti, vphi_tt, log_theta, theta_w_, noise_var, w0, wbar00,
                        lambda_dag, t)
            tmp = c_t[1] * c_t[0] + c_t[2]
            # calculate xi_t
            if tmp != 0:
                xi_t = (c_t[2] * c_t[0] + c_t[3]) / tmp
            else:
                xi_t = 1
        else:
            c_t = np.zeros(4, dtype=np.float64)
            c_t[1] = noise_var * w0 + vphi_tt[t, t] * wbar00
        # print("xi",xi_t)
        if xi_t < 0 or np.isnan(xi_t):
            # print(vphi_tt[t,t])
            # print(np.amin(vphi_tt))
            break
        # calculate p_ti[t] & epsilon_t
        log_theta[t] = np.log(xi_t)
        p_ti[t] = xi_t * w0
        epsilon_t = p_ti[t] + w0 * c_t[0]  # (19c)
        # calculate v_gamma
        v_gamma = max((c_t[1] * (xi_t ** 2) - 2 * c_t[2] * xi_t + c_t[3]) / (epsilon_t ** 2), 1e-100)
        # if v_gamma<0:
        #     break
        # print("\nv_gamma",v_gamma,"xi",xi_t)
        # calculate mean rt
        rhat_t = xi_t * zt[:, t].reshape(M, 1) + theta_t * (lambda_dag * rhat_t - AAH @ rhat_t)  # (16)
        memory = np.zeros((N, 1), dtype=np.float64)
        for i in range(t + 1):  # sum:0 ~ t
            memory += p_ti[i] * xt[:, i].reshape(N, 1)
        rt = (AH @ rhat_t + memory) / epsilon_t

        ''' NLE Post_Demodulation '''
        xhat_t, vhat_t, xt_, _ = NLE(v_gamma, rt, mu=mu, orth=orth)
        # print("vhat",vhat_t)
        # vhat_t = max(vhat_t,1e-18)
        MSE[t] = np.mean((x - xhat_t) ** 2)
        # print("iteration:",t,"MSE",10*np.log10(MSE[t]))
        VAR[t] = vhat_t
        ct = 2
        if t == T - 1:
            # print("xhat_t",xhat_t)
            break
        elif t >= ct:
            thre1 = 1e-6  # stop damping
            thre2 = 1e-7  # stop the algorithm
            comp = max(abs(vhat_t - VAR[t - ct:t]))  # reflect the change of MSE
            if thre2 < comp <= thre1:
                damping_flag = False
            elif comp <= thre2:
                MSE[t + 1:T] = MSE[t]
                VAR[t + 1:T] = VAR[t]
                break

        # TODO:change form to avoid overflow
        xt[:, t + 1] = xt_.T
        zt[:, t + 1] = (y - A @ xt[:, t + 1].reshape(N, 1)).T
        for tt in range(t + 2):  # error covariance matrix
            vphi_tt[t + 1, tt] = (zt[:, t + 1].T @ zt[:, tt] / N - delta * noise_var) / w0
            if orth is False:
                vphi_tt[t + 1, tt] = max(vphi_tt[t + 1, tt], 1e-10)
            # for correlated
            # vphi_tt[t+1,tt] = (zt[:,t+1].T@zt[:,tt]/N - delta*noise_var)/w0
            vphi_tt[tt, t + 1] = vphi_tt[t + 1, tt]

        # if np.amin(vphi_tt) < 0:
        #     print("enough")
        #     MSE[t+1:T] = MSE[t]
        #     break

        """ damping """
        lt = min(L, t + 2)
        # calculate Vtilde (24a)
        Vtilde = np.zeros((lt, lt), dtype=np.float64)
        Vtilde[:, :] = vphi_tt[t - lt + 2:t + 2, t - lt + 2:t + 2]  # t-lt+2~t+1
        if damping_flag is False or min(la.eigvalsh(Vtilde)) <= 0: \
                # or np.amin(vphi_tt) < 0:  # not positive semi-definite
            # print("no damping")
            if vphi_tt[t + 1, t + 1] > vphi_tt[
                t, t]:  # or np.amin(vphi_tt) < 0:  # keep the last value as it's not better
                # print("last value")
                xt[:, t + 1] = xt[:, t]
                vphi_tt[t + 1, t + 1] = vphi_tt[t, t]
                vphi_tt[0:t + 1, t + 1] = vphi_tt[0:t + 1, t]
                vphi_tt[t + 1, 0:t + 1] = vphi_tt[t, 0:t + 1]
                zt[:, t + 1] = zt[:, t]
        else:
            """TODO: calculate det(Vtilde)---
            do not use det to decide whether a matrix is singular"""
            f = 0
            while 1 / la.cond(Vtilde, 1) < 1e-15:  # use the reciprocal of l1-norm to confirm singularity of a matrix.
                # print("singular")
                f = f + 1
                if t - lt + 2 - f < 0:
                    singular = True
                    break
                Vtilde[:lt - 1, :lt - 1] = vphi_tt[t - lt + 2 - f:t + 1 - f, t - lt + 2 - f:t + 1 - f]
            if singular:
                break
            # if 1/la.cond(Vtilde,1) < 1e-15:
            #     break
            # calculate the damping vector zeta_t (31)(32)
            zeta_t = la.inv(Vtilde) @ np.ones((lt, 1), dtype=np.float64)
            v_phi = 1 / (np.ones(lt) @ zeta_t)
            zeta_t = v_phi * zeta_t
            # print("v_phi",v_phi)
            # damping the others
            # xt & zt
            xt[:, t + 1] = zeta_t[lt - 1] * xt[:, t + 1]
            zt[:, t + 1] = zeta_t[lt - 1] * zt[:, t + 1]
            for i in range(lt - 1):
                xt[:, t + 1] += zeta_t[i] * xt[:, t - lt + 2 - f + i]
                zt[:, t + 1] += zeta_t[i] * zt[:, t - lt + 2 - f + i]
            # vphi_tt
            vphi_tt[t + 1, t + 1] = max(v_phi, 1e-10)
            for tt in range(t + 1):
                vphi_tt[t + 1, tt] = zeta_t[lt - 1] * vphi_tt[t + 1, tt]
                for i in range(lt - 1):
                    vphi_tt[t + 1, tt] += zeta_t[i] * vphi_tt[t - lt + 2 - f + i, tt]
                vphi_tt[t + 1, tt] = max(vphi_tt[t + 1, tt], 1e-9)
                vphi_tt[tt, t + 1] = vphi_tt[t + 1, tt]
            # print("v_phi",vphi_tt[t+1,t+1])

    return xhat_t, MSE


# theta_(i) * w(j)——i>=1, j>=0
def theta_w(lambda_dag, B, sign, log_B, log_theta_i, j, N):
    tmp = (lambda_dag - B) * sign ** j * np.exp(log_theta_i + j * log_B)
    res = 1 / N * sum(tmp)
    return res


# calculate c_ti:0<=i<=3 (36)
def Get_c(p_ti, vphi_tt, log_theta, theta_w_, noise_var, w0, wbar00, lambda_dag, t):
    c_t = np.zeros(4, dtype=np.float64)
    # c0 
    c_t[0] = sum(p_ti[0:t]) / w0
    # c1
    c_t[1] = noise_var * w0 + vphi_tt[t, t] * wbar00
    # c2
    term1 = np.zeros(t)
    term1[0:t] = p_ti[0:t]
    coef1 = noise_var + vphi_tt[t, 0:t] * (lambda_dag - w0)
    term2 = np.zeros(t)
    term2[0] = theta_w_[t]
    term2[1:t] = p_ti[0:t - 1] * np.exp(log_theta[1:t] - log_theta[0:t - 1])
    c_t[2] = sum(vphi_tt[t, 0:t] * term2 - coef1 * term1)
    # c3
    for i in range(t):
        for j in range(t):
            if 2 * t - i - j < t + 1:
                coef1 = np.exp(log_theta[i] + log_theta[j] - log_theta[i + j - t])
            elif 2 * t - i - j == t + 1:
                coef1 = np.exp(log_theta[i] + log_theta[j] - log_theta[0])
            else:
                coef1 = np.exp(log_theta[i] + log_theta[j] - log_theta[0] - log_theta[i + j + 1])
            term1 = (noise_var + vphi_tt[i, j] * lambda_dag) * coef1 * theta_w_[2 * t - i - j - 1]
            if 2 * t - i - j + 1 < t + 1:
                coef2 = np.exp(log_theta[i] + log_theta[j] - log_theta[i + j - t - 1])
            elif 2 * t - i - j + 1 == t + 1:
                coef2 = np.exp(log_theta[i] + log_theta[j] - log_theta[0])
            else:
                coef2 = np.exp(log_theta[i] + log_theta[j] - log_theta[0] - log_theta[i + j])
            term2 = vphi_tt[i, j] * coef2 * theta_w_[2 * t - i - j]
            term3 = vphi_tt[i, j] * p_ti[i] * p_ti[j]
            c_t[3] += (term1 - term2 - term3)
    return c_t
