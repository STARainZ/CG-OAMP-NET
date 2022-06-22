#!/usr/bin/python
from __future__ import division
from __future__ import print_function
from .train import load_trainable_vars, save_trainable_vars
from .MIMO_detection import sample_gen
import numpy as np
import sys
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
# import tensorflow as tf

sq2 = np.sqrt(2)
sq10 = np.sqrt(10)
sq42 = np.sqrt(42)


def nle(mu, mean, var, thre):
    # ext_probs = np.zeros((ule.shape[0], 2 ** (mu // 2)))
    if mu == 2:  # {-1,+1}
        P0 = tf.maximum(tf.exp(-tf.square(-1 / sq2 - mean) / (2 * var)), thre)  # (bs, 2N, 1)
        P1 = tf.maximum(tf.exp(-tf.square(1 / sq2 - mean) / (2 * var)), thre)
        u_post = (P1 - P0) / (P1 + P0) / sq2
    elif mu == 4:  # {-3,-1,+1,+3}
        P_3 = tf.maximum(tf.exp(-tf.square(-3 / sq10 - mean) / (2 * var)), thre)
        P_1 = tf.maximum(tf.exp(-tf.square(-1 / sq10 - mean) / (2 * var)), thre)
        P1 = tf.maximum(tf.exp(-tf.square(1 / sq10 - mean) / (2 * var)), thre)
        P3 = tf.maximum(tf.exp(-tf.square(3 / sq10 - mean) / (2 * var)), thre)
        u_post = (-3 * P_3 - P_1 + P1 + 3 * P3) / (P_3 + P_1 + P1 + P3) / sq10
    else:  # {-1,+1}
        P_7 = tf.maximum(tf.exp(-tf.square(-7 / sq42 - mean) / (2 * var)), thre)
        P_5 = tf.maximum(tf.exp(-tf.square(-5 / sq42 - mean) / (2 * var)), thre)
        P_3 = tf.maximum(tf.exp(-tf.square(-3 / sq42 - mean) / (2 * var)), thre)
        P_1 = tf.maximum(tf.exp(-tf.square(-1 / sq42 - mean) / (2 * var)), thre)
        P1 = tf.maximum(tf.exp(-tf.square(1 / sq42 - mean) / (2 * var)), thre)
        P3 = tf.maximum(tf.exp(-tf.square(3 / sq42 - mean) / (2 * var)), thre)
        P5 = tf.maximum(tf.exp(-tf.square(5 / sq42 - mean) / (2 * var)), thre)
        P7 = tf.maximum(tf.exp(-tf.square(7 / sq42 - mean) / (2 * var)), thre)
        u_post = (-7 * P_7 - 5 * P_5 - 3 * P_3 - P_1 + P1 + 3 * P3 + 5 * P5 + 7 * P7) / (
                P_7 + P_5 + P_3 + P_1 + P1 + P3 + P5 + P7) / sq42

    return u_post


def CG(u, p, residual, r_norm, XI, sample_size):
    # compute the approximate solution based on prior conjugate direction and residual
    XI_p = tf.matmul(XI, p)  # bs*2M*1
    a = r_norm / tf.matmul(tf.transpose(p, perm=[0, 2, 1]), XI_p)
    u = tf.add(u, a * p)
    # compute conjugate direction and residual
    residual = tf.add(residual, -a * XI_p)
    r_norm_last = r_norm
    r_norm = tf.reshape(tf.square(tf.norm(residual, axis=(1, 2))), [sample_size, 1, 1])
    # r_norm_last = tf.maximum(r_norm_last,tf.constant(1e-20))
    b = r_norm / r_norm_last
    p = tf.add(residual, b * p)
    # r_norm = tf.maximum(r_norm, tf.constant(1e-20))
    return u, p, residual, r_norm


def build_CG_OAMP(trainSet):
    T, I = trainSet.T, trainSet.icg
    use_OFDM, K, CP, CP_flag = trainSet.use_OFDM, trainSet.K, trainSet.CP, trainSet.CP_flag
    if use_OFDM:
        Mr, Nt = trainSet.Mr * K, trainSet.Nt * K
    else:
        Mr, Nt = trainSet.Mr, trainSet.Nt
    mu, SNR = trainSet.mu, trainSet.snr
    version = trainSet.version
    lr, maxit = trainSet.lr, trainSet.maxit
    vsample_size = trainSet.vsample_size
    total_batch, batch_size = trainSet.total_batch, trainSet.batch_size
    savefile = trainSet.savefile
    prob, test = trainSet.prob, trainSet.test
    layers = []  # layerinfo:(name,xhat_,newvars)

    H_ = prob.H_  # 2M*2N
    x_ = prob.x_  # bs*2N*1
    y_ = prob.y_  # bs*2M*1
    sigma2_ = prob.sigma2_  # bs*1*1
    sample_size = prob.sample_size_  # bs
    eigvalue = prob.eigvalue_  # M*1

    HT_ = tf.transpose(H_, perm=[0, 2, 1])
    HHT = tf.matmul(H_, HT_)
    OneOver_trHTH = tf.reshape(1 / tf.trace(tf.matmul(HT_, H_)), [sample_size, 1, 1])
    sigma2_I = sigma2_ / 2 * tf.eye(2 * Mr, batch_shape=[sample_size], dtype=tf.float32)

    # precompute some tensorflow constants
    epsilon = tf.constant(1e-10, dtype=tf.float32)
    rth = tf.constant(1e-4, dtype=tf.float64)
    pth = tf.constant(1e-100, dtype=tf.float64)
    v_sqr_last = tf.constant(0, dtype=tf.float32)
    x_hat = tf.zeros_like(x_, dtype=tf.float32)

    for t in range(T):
        theta_ = tf.Variable(float(1), dtype=tf.float32, name='theta_' + str(t))
        gamma_ = tf.Variable(float(1), dtype=tf.float32, name='gamma_' + str(t))
        beta_ = tf.Variable(float(0.5), dtype=tf.float32, name='beta_' + str(t))
        if version == 1:
            phi_ = tf.Variable(float(1), dtype=tf.float32, name='phi_' + str(t))
            xi_ = tf.Variable(float(0), dtype=tf.float32, name='xi_' + str(t))

        p_noise = y_ - tf.matmul(H_, x_hat)  # bs*2M*1
        v_sqr = (tf.reshape(tf.square(tf.norm(p_noise, axis=(1, 2))),
                            [sample_size, 1, 1]) - Mr * sigma2_) * OneOver_trHTH  # bs*1*1
        v_sqr = beta_ * v_sqr + (1 - beta_) * v_sqr_last
        v_sqr = tf.maximum(v_sqr, epsilon)
        v_sqr_last = v_sqr

        # with tf.device("/cpu:0"):
        # CG for the calculation of u=(xi)**-1@p_noise       
        XI = tf.cast(HHT + sigma2_I / v_sqr, dtype=tf.float64)  # 2M*2M
        # initial value
        u = tf.zeros_like(y_, dtype=tf.float64)
        residual = tf.cast(p_noise, dtype=tf.float64)
        p = residual  # bs*2M*1
        r_norm = tf.reshape(tf.square(tf.norm(residual, axis=(1, 2))), [sample_size, 1, 1])  # bs*1*1
        # build CG into a while_loop         
        i_, u_, _, _, r_norm_ = tf.while_loop(
            cond=lambda i, u, p, residual, r_norm: tf.reduce_any(tf.greater(r_norm, rth)),
            body=lambda i, u, p, residual, r_norm: (i + 1, *CG(u, p, residual, r_norm, XI, sample_size)),
            loop_vars=(tf.constant(0), u, p, residual, r_norm), maximum_iterations=I)
        u_ = tf.cast(u_, dtype=tf.float32)

        estimate = tf.reduce_mean(eigvalue / tf.add(eigvalue, sigma2_ / 2 / v_sqr),
                                  1) * 2 * Mr  # add and divide support broadcasting bs*1
        nor_coef = 2 * Nt / tf.reshape(estimate, [sample_size, 1, 1])  # reshape bs*1*1
        r = x_hat + gamma_ * nor_coef * tf.matmul(HT_, u_)
        tau_sqr = v_sqr * ((theta_ ** 2) * nor_coef - 2 * theta_ + 1)  # bs*1*1
        tau_sqr = tf.cast(tf.maximum(tau_sqr, epsilon), dtype=tf.float64)
        r = tf.cast(r, dtype=tf.float64)
        x_hat = nle(mu, r, tau_sqr, pth)

        x_hat = tf.cast(x_hat, dtype=tf.float32)
        r = tf.cast(r, dtype=tf.float32)
        if version == 1:
            x_hat = phi_ * (x_hat - xi_ * r)

        if version == 0:
            layers.append(('CG-OAMP T={0}'.format(t), x_hat, (theta_, gamma_,)))
        else:
            layers.append(('CG-OAMP T={0}'.format(t), x_hat, (theta_, gamma_, beta_, phi_, xi_,)))

    loss_ = tf.nn.l2_loss(x_hat - x_)
    lr_ = tf.Variable(lr, name='lr', trainable=False)
    if tf.trainable_variables() is not None:
        train = tf.train.AdamOptimizer(lr_).minimize(loss_, var_list=tf.trainable_variables())
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    state = load_trainable_vars(sess, savefile)
    done = state.get('done', [])
    log = str(state.get('log', ''))

    for name, _, var_list in layers:
        if len(var_list):
            describe_var_list = 'extending ' + ','.join([v.name for v in var_list])
        else:
            describe_var_list = 'fine tuning all ' + \
                                ','.join([v.name for v in tf.trainable_variables()])
        done = np.append(done, name)
        print(name + ' ' + describe_var_list)
    print(log)

    if test:
        return sess, x_hat

    loss_history = []
    save = {}  # for the best model
    ivl = 1

    if use_OFDM:
        from .MIMO_OFDM_detection import sample_gen_MIMO_OFDM
        yval, xval, Hval, sigma2val, eigval = sample_gen_MIMO_OFDM(trainSet, vsample_size,
                                                                   training_flag=False)
    else:
        _, _, _, _, _, yval, xval, Hval, sigma2val, eigval = sample_gen(trainSet, 1, vsample_size)
    run_opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    for i in range(maxit + 1):
        if use_OFDM:
            # for MIMO-OFDM model of all subcarriers
            y, x, H, sigma2, eig = sample_gen_MIMO_OFDM(trainSet, batch_size * total_batch, training_flag=True)
        else:
            y, x, H, sigma2, eig, _, _, _, _, _ = sample_gen(trainSet, batch_size * total_batch, 1)
        # TODO:shuffling after every epoch -- not easy -- the order of each elements should match
        if i % ivl == 0:  # validation:don't use optimizer
            loss = sess.run(loss_, feed_dict={prob.y_: yval,
                                              prob.x_: xval, prob.H_: Hval, prob.sigma2_: sigma2val,
                                              prob.sample_size_: vsample_size,
                                              prob.eigvalue_: eigval}, options=run_opts)
            if np.isnan(loss):
                raise RuntimeError('loss is NaN')

            loss_history = np.append(loss_history, loss)
            loss_best = loss_history.min()
            # for the best model
            # TODO:change back to early stopping
            if loss == loss_best:
                for v in tf.trainable_variables():
                    save[str(v.name)] = sess.run(v)
            sys.stdout.write('\ri={i:<6d} loss={loss:.9f} (best={best:.9f})'
                             .format(i=i, loss=loss, best=loss_best))
            sys.stdout.flush()
            if i % (100 * ivl) == 0:
                print('')

        for m in range(total_batch):
            sess.run(train, feed_dict={prob.y_: y[m * batch_size:(m + 1) * batch_size],
                                       prob.x_: x[m * batch_size:(m + 1) * batch_size],
                                       prob.H_: H[m * batch_size:(m + 1) * batch_size],
                                       prob.sigma2_: sigma2[m * batch_size:(m + 1) * batch_size],
                                       prob.sample_size_: batch_size,
                                       prob.eigvalue_: eig[m * batch_size:(m + 1) * batch_size]},
                     options=run_opts)

    # for the best model----it's for the strange phenomenon
    tv = dict([(str(v.name), v) for v in tf.trainable_variables()])
    for k, d in save.items():
        if k in tv:
            sess.run(tf.assign(tv[k], d))
            print('restoring ' + k + ' = ' + str(d))

    log = log + '\nloss={loss:.9f} in {i} iterations   best={best:.9f} in {j} ' \
                'iterations'.format(loss=loss, i=i, best=loss_best, j=loss_history.argmin())

    state['done'] = done
    state['log'] = log
    save_trainable_vars(sess, savefile, **state)

    return sess, x_hat


def build_OAMP(trainSet):
    T, I = trainSet.T, trainSet.icg
    use_OFDM, K, CP, CP_flag = trainSet.use_OFDM, trainSet.K, trainSet.CP, trainSet.CP_flag
    if use_OFDM:
        Mr, Nt = trainSet.Mr * K, trainSet.Nt * K
    else:
        Mr, Nt = trainSet.Mr, trainSet.Nt
    mu, SNR = trainSet.mu, trainSet.snr
    version = trainSet.version
    lr, maxit = trainSet.lr, trainSet.maxit
    vsample_size = trainSet.vsample_size
    total_batch, batch_size = trainSet.total_batch, trainSet.batch_size
    savefile = trainSet.savefile
    prob, test = trainSet.prob, trainSet.test
    layers = []  # layerinfo:(name,xhat_,newvars)

    H_ = prob.H_
    x_ = prob.x_
    y_ = prob.y_
    sigma2_ = prob.sigma2_
    sample_size = prob.sample_size_

    # precompute some tensorflow constants
    epsilon = tf.constant(1e-10, dtype=tf.float32)
    pth = tf.constant(1e-100, dtype=tf.float64)
    HT_ = tf.transpose(H_, perm=[0, 2, 1])
    HHT = tf.matmul(H_, HT_)
    OneOver_trHTH = tf.reshape(1 / tf.trace(tf.matmul(HT_, H_)), [sample_size, 1, 1])
    sigma2_I = sigma2_ / 2 * tf.eye(2 * Mr, batch_shape=[sample_size], dtype=tf.float32)

    v_sqr_last = tf.constant(0, dtype=tf.float32)
    x_hat = tf.zeros_like(x_, dtype=tf.float32)

    for t in range(T):
        theta_ = tf.Variable(float(1), dtype=tf.float32, name='theta_' + str(t))
        gamma_ = tf.Variable(float(1), dtype=tf.float32, name='gamma_' + str(t))
        if version == 1:
            phi_ = tf.Variable(float(1), dtype=tf.float32, name='phi_' + str(t))
            xi_ = tf.Variable(float(0), dtype=tf.float32, name='xi_' + str(t))
        p_noise = y_ - tf.matmul(H_, x_hat)
        v_sqr = (tf.reshape(tf.square(tf.norm(p_noise, axis=(1, 2))),
                            [sample_size, 1, 1]) - Mr * sigma2_) * OneOver_trHTH
        v_sqr = 0.5 * v_sqr + 0.5 * v_sqr_last
        v_sqr = tf.maximum(v_sqr, epsilon)
        v_sqr_last = v_sqr

        # with tf.device("/cpu:0"):
        w_hat = tf.matmul(HT_, tf.linalg.inv(HHT + sigma2_I / v_sqr))
        nor_coef = 2 * Nt / tf.reshape(tf.trace(tf.matmul(w_hat, H_)), [sample_size, 1, 1])
        r = x_hat + gamma_ * nor_coef * tf.matmul(w_hat, p_noise)

        tau_sqr = v_sqr * ((theta_ ** 2) * nor_coef - 2 * theta_ + 1)
        tau_sqr = tf.cast(tf.maximum(tau_sqr, epsilon), dtype=tf.float64)
        r = tf.cast(r, dtype=tf.float64)
        x_hat = nle(mu, r, tau_sqr, pth)

        x_hat = tf.cast(x_hat, dtype=tf.float32)
        r = tf.cast(r, dtype=tf.float32)
        if version == 1:
            x_hat = phi_ * (x_hat - xi_ * r)  # (18)

        if version == 0:
            layers.append(('OAMP T={0}'.format(t), x_hat, (theta_, gamma_,)))
        else:
            layers.append(('OAMP T={0}'.format(t), x_hat, (theta_, gamma_, phi_, xi_,)))

    loss_ = tf.nn.l2_loss(x_hat - x_)
    lr_ = tf.Variable(lr, name='lr', trainable=False)
    if tf.trainable_variables() is not None:
        train = tf.train.AdamOptimizer(lr_).minimize(loss_, var_list=tf.trainable_variables())
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    state = load_trainable_vars(sess, savefile)
    done = state.get('done', [])
    log = str(state.get('log', ''))

    for name, _, var_list in layers:
        if len(var_list):
            describe_var_list = 'extending ' + ','.join([v.name for v in var_list])
        else:
            describe_var_list = 'fine tuning all ' + ','.join([v.name for v in tf.trainable_variables()])
        done = np.append(done, name)
        print(name + ' ' + describe_var_list)
    print(log)

    if test:
        return sess, x_hat

    loss_history = []
    save = {}  # for the best model
    ivl = 1

    if use_OFDM:
        from .MIMO_OFDM_detection import sample_gen_MIMO_OFDM
        yval, xval, Hval, sigma2val = sample_gen_MIMO_OFDM(trainSet, vsample_size,
                                                           training_flag=False)
    else:
        _, _, _, _, yval, xval, Hval, sigma2val = sample_gen(trainSet, 1, vsample_size)
    for i in range(maxit + 1):
        if use_OFDM:
            # for MIMO-OFDM model of all subcarriers
            y, x, H, sigma2 = sample_gen_MIMO_OFDM(trainSet, batch_size * total_batch, training_flag=True)
        else:
            y, x, H, sigma2, _, _, _, _ = sample_gen(trainSet, batch_size * total_batch, 1)

        if i % ivl == 0:  # validation:don't use optimizer
            loss = sess.run(loss_, feed_dict={prob.y_: yval,
                                              prob.x_: xval, prob.H_: Hval, prob.sigma2_: sigma2val,
                                              prob.sample_size_: vsample_size})  # 1000 samples and labels
            if np.isnan(loss):
                raise RuntimeError('loss is NaN')
            loss_history = np.append(loss_history, loss)
            loss_best = loss_history.min()
            # for the best model
            if loss == loss_best:
                for v in tf.trainable_variables():
                    save[str(v.name)] = sess.run(v)
                    #
            sys.stdout.write('\ri={i:<6d} loss={loss:.9f} (best={best:.9f})'
                             .format(i=i, loss=loss, best=loss_best))
            sys.stdout.flush()
            if i % (100 * ivl) == 0:
                print('')

        for m in range(total_batch):
            sess.run(train, feed_dict={prob.y_: y[m * batch_size:(m + 1) * batch_size],
                                       prob.x_: x[m * batch_size:(m + 1) * batch_size],
                                       prob.H_: H[m * batch_size:(m + 1) * batch_size],
                                       prob.sigma2_: sigma2[m * batch_size:(m + 1) * batch_size],
                                       prob.sample_size_: batch_size})  # 1000 samples and labels
    # for the best model----it's for the strange phenomenon
    tv = dict([(str(v.name), v) for v in tf.trainable_variables()])
    for k, d in save.items():
        if k in tv:
            sess.run(tf.assign(tv[k], d))
            print('restoring ' + k + ' = ' + str(d))

    log = log + '\nloss={loss:.9f} in {i} iterations   best={best:.9f} in {j} ' \
                'iterations'.format(loss=loss, i=i, best=loss_best, j=loss_history.argmin())

    state['done'] = done
    state['log'] = log
    save_trainable_vars(sess, savefile, **state)

    return sess, x_hat
