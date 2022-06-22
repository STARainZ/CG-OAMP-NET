#!/usr/bin/python
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


# import tensorflow as tf


def save_trainable_vars(sess, filename, **kwargs):
    """save a .npz archive in `filename`  with
    the current value of each variable in tf.trainable_variables()
    plus any keyword numpy arrays.
    """
    save = {}
    for v in tf.trainable_variables():
        save[str(v.name)] = sess.run(v)
    save.update(kwargs)
    np.savez(filename, **save)


def load_trainable_vars(sess, filename):
    """load a .npz archive and assign the value of each loaded
    ndarray to the trainable variable whose name matches the
    archive key.  Any elements in the archive that do not have
    a corresponding trainable variable will be returned in a dict.
    """
    other = {}
    try:
        tv = dict([(str(v.name), v) for v in tf.trainable_variables()])
        for k, d in np.load(filename).items():
            if k in tv:
                print('restoring ' + k)
                sess.run(tf.assign(tv[k], d))
            else:
                other[k] = d
    except IOError:
        pass
    return other
