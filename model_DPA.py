#! /usr/bin/python
# -*- coding: utf8 -*-

import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import *
from config import config, log_config

batch_size = config.TRAIN.batch_size


def T1(y_1, is_train=False, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("t1", reuse=reuse):
        n_1 = InputLayer(y_1)

        t1 = Conv2d(n_1, 256, (3, 3), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                    name='init256/1')

        t1 = bottleneck(t1, is_train=is_train, con_number=256, block_num=1)
        t1_a1 = t1
        t1 = SubpixelConv2d(t1, scale=2, n_out_channel=None, act=None, name='pixelshufflerx2/1')
        t1 = Conv2d(t1, 256, (3, 3), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                    name='b_residual1/n16/2')

        t1 = bottleneck(t1, is_train=is_train, con_number=256, block_num=2)
        t1_a2 = t1
        t1 = SubpixelConv2d(t1, scale=2, n_out_channel=None, act=None, name='pixelshufflerx2/2')
        t1 = Conv2d(t1, 256, (3, 3), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                    name='b_residual2/n16/2')

        t1 = bottleneck(t1, is_train=is_train, con_number=256, block_num=3)
        t1_a3 = t1
        t1 = SubpixelConv2d(t1, scale=2, n_out_channel=None, act=None, name='pixelshufflerx2/3')
        t1 = Conv2d(t1, 256, (3, 3), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                    name='b_residual3/n16/2')

        t1 = bottleneck(t1, is_train=is_train, con_number=256, block_num=4)
        t1_a4 = t1
        t1 = SubpixelConv2d(t1, scale=2, n_out_channel=None, act=None, name='pixelshufflerx2/4')
        t1 = Conv2d(t1, 32, (3, 3), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                    name='global_n32/s1')

        t1 = Conv2d(t1, 1, (3, 3), (1, 1), act=tl.act.hard_tanh, padding='SAME', W_init=w_init, b_init=b_init, name='out')
        return t1, t1_a1, t1_a2, t1_a3, t1_a4


def T2(y_2, t2_a1, t2_a2, t2_a3, t2_a4, is_train=False, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    j = 1
    with tf.variable_scope("t2", reuse=reuse):
        n_2 = InputLayer(y_2)

        t2 = Conv2d(n_2, 256, (3, 3), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                    name='init256/1')

        temp_global = t2
        temp = t2
        t2 = block(t2, 5, 64, is_train=is_train, idx_=11, task=2)
        t2 = Conv2d(t2, 256, (1, 1), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                    name='rdb1/n256/1')
        t2 = ElementwiseLayer([t2, temp], combine_fn=tf.add, name='b_residual1/add1')
        t2, spa1 = attention(t2, t2_a1, con_number=256, block_num=1)
        temp = t2
        t2 = block(t2, 2, 64, is_train=is_train, idx_=12, task=2)
        t2 = Conv2d(t2, 256, (1, 1), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                    name='rdb1/n256/2')
        t2 = ElementwiseLayer([t2, temp], combine_fn=tf.add, name='b_residual1/add2')
        t2 = ElementwiseLayer([t2, temp_global], combine_fn=tf.add, name='b_residual1/add3')
        t2 = SubpixelConv2d(t2, scale=2, n_out_channel=None, act=tf.nn.selu, name='attention1/pixelshufflerx2/1')
        t2 = Conv2d(t2, 256, (3, 3), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                    name='b_residual1/n16/2')

        temp_global = t2
        temp = t2
        t2 = block(t2, 5, 64, is_train=is_train, idx_=21, task=2)
        t2 = Conv2d(t2, 256, (1, 1), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                    name='rdb2/n256/1')
        t2 = ElementwiseLayer([t2, temp], combine_fn=tf.add, name='b_residual2/add1')
        t2, spa2 = attention(t2, t2_a2, con_number=256, block_num=2)
        temp = t2
        t2 = block(t2, 2, 64, is_train=is_train, idx_=22, task=2)
        t2 = Conv2d(t2, 256, (1, 1), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                    name='rdb2/n256/2')
        t2 = ElementwiseLayer([t2, temp], combine_fn=tf.add, name='b_residual2/add2')
        t2 = ElementwiseLayer([t2, temp_global], combine_fn=tf.add, name='b_residual2/add3')
        t2 = SubpixelConv2d(t2, scale=2, n_out_channel=None, act=tf.nn.selu, name='attention2/pixelshufflerx2/1')
        # t2 = BatchNormLayer(t2, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='attention2/b2')
        t2 = Conv2d(t2, 256, (3, 3), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                    name='b_residual2/n16/2')

        temp_global = t2
        temp = t2
        t2 = block(t2, 5, 64, is_train=is_train, idx_=31, task=2)
        t2 = Conv2d(t2, 256, (1, 1), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                    name='rdb3/n256/1')
        t2 = ElementwiseLayer([t2, temp], combine_fn=tf.add, name='b_residual3/add1')
        t2, spa3 = attention(t2, t2_a3, con_number=256, block_num=3)
        temp = t2
        t2 = block(t2, 2, 64, is_train=is_train, idx_=32, task=2)
        t2 = Conv2d(t2, 256, (1, 1), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                    name='rdb3/n256/2')
        t2 = ElementwiseLayer([t2, temp], combine_fn=tf.add, name='b_residual3/add2')
        t2 = ElementwiseLayer([t2, temp_global], combine_fn=tf.add, name='b_residual3/add3')
        t2 = SubpixelConv2d(t2, scale=2, n_out_channel=None, act=tf.nn.selu, name='attention3/pixelshufflerx2/1')
        # t2 = BatchNormLayer(t2, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='attention3/b2')
        t2 = Conv2d(t2, 256, (3, 3), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                    name='b_residual3/n16/2')

        temp_global = t2
        temp = t2
        t2 = block(t2, 5, 64, is_train=is_train, idx_=41, task=2)
        t2 = Conv2d(t2, 256, (1, 1), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                    name='rdb4/n256/1')
        t2 = ElementwiseLayer([t2, temp], combine_fn=tf.add, name='b_residual4/add1')
        t2, spa4 = attention(t2, t2_a4, con_number=256, block_num=4)
        temp = t2
        t2 = block(t2, 2, 64, is_train=is_train, idx_=42, task=2)
        t2 = Conv2d(t2, 256, (1, 1), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                    name='rdb4/n256/2')
        t2 = ElementwiseLayer([t2, temp], combine_fn=tf.add, name='b_residual4/add2')
        t2 = ElementwiseLayer([t2, temp_global], combine_fn=tf.add, name='b_residual4/add3')
        t2 = SubpixelConv2d(t2, scale=2, n_out_channel=None, act=tf.nn.selu, name='attention4/pixelshufflerx2/1')
        t2 = Conv2d(t2, 32, (3, 3), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                    name='global_n32/s1')
        t2 = Conv2d(t2, 1, (3, 3), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init, name='t2/out')
        return t2



def T2_compare(y_2, is_train=False, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = tf.constant_initializer(value=0.0)

    with tf.variable_scope("t2", reuse=reuse):
        n_2 = InputLayer(y_2)

        t2 = Conv2d(n_2, 256, (3, 3), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                    name='init256/1')

        temp_global = t2
        temp = t2
        t2 = block(t2, 3, 64, is_train=is_train, idx_=11, task=2)
        t2 = Conv2d(t2, 256, (3, 3), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                    name='rdb1/n256/1')
        t2 = ElementwiseLayer([t2, temp], combine_fn=tf.add, name='b_residual1/add1')
        temp = t2
        t2 = block(t2, 3, 64, is_train=is_train, idx_=12, task=2)
        t2 = Conv2d(t2, 256, (3, 3), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                    name='rdb1/n256/2')
        t2 = ElementwiseLayer([t2, temp], combine_fn=tf.add, name='b_residual1/add2')
        t2 = ElementwiseLayer([t2, temp_global], combine_fn=tf.add, name='b_residual1/add3')
        t2 = SubpixelConv2d(t2, scale=2, n_out_channel=None, act=tf.nn.selu, name='attention1/pixelshufflerx2/1')
        t2 = Conv2d(t2, 256, (3, 3), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                    name='b_residual1/n16/2')

        temp_global = t2
        temp = t2
        t2 = block(t2, 3, 64, is_train=is_train, idx_=21, task=2)
        t2 = Conv2d(t2, 256, (3, 3), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                    name='rdb2/n256/1')
        t2 = ElementwiseLayer([t2, temp], combine_fn=tf.add, name='b_residual2/add1')
        temp = t2
        t2 = block(t2, 3, 64, is_train=is_train, idx_=22, task=2)
        t2 = Conv2d(t2, 256, (3, 3), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                    name='rdb2/n256/2')
        t2 = ElementwiseLayer([t2, temp], combine_fn=tf.add, name='b_residual2/add2')
        t2 = ElementwiseLayer([t2, temp_global], combine_fn=tf.add, name='b_residual2/add3')
        t2 = SubpixelConv2d(t2, scale=2, n_out_channel=None, act=tf.nn.selu, name='attention2/pixelshufflerx2/1')
        t2 = Conv2d(t2, 256, (3, 3), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                    name='b_residual2/n16/2')

        temp_global = t2
        temp = t2
        t2 = block(t2, 3, 64, is_train=is_train, idx_=31, task=2)
        t2 = Conv2d(t2, 256, (3, 3), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                    name='rdb3/n256/1')
        t2 = ElementwiseLayer([t2, temp], combine_fn=tf.add, name='b_residual3/add1')
        temp = t2
        t2 = block(t2, 3, 64, is_train=is_train, idx_=32, task=2)
        t2 = Conv2d(t2, 256, (3, 3), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                    name='rdb3/n256/2')
        t2 = ElementwiseLayer([t2, temp], combine_fn=tf.add, name='b_residual3/add2')
        t2 = ElementwiseLayer([t2, temp_global], combine_fn=tf.add, name='b_residual3/add3')
        t2 = SubpixelConv2d(t2, scale=2, n_out_channel=None, act=tf.nn.selu, name='attention3/pixelshufflerx2/1')
        t2 = Conv2d(t2, 256, (3, 3), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                    name='b_residual3/n16/2')

        temp_global = t2
        temp = t2
        t2 = block(t2, 3, 64, is_train=is_train, idx_=41, task=2)
        t2 = Conv2d(t2, 256, (3, 3), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                    name='rdb4/n256/1')
        t2 = ElementwiseLayer([t2, temp], combine_fn=tf.add, name='b_residual4/add1')
        temp = t2
        t2 = block(t2, 3, 64, is_train=is_train, idx_=42, task=2)
        t2 = Conv2d(t2, 256, (3, 3), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                    name='rdb4/n256/2')
        t2 = ElementwiseLayer([t2, temp], combine_fn=tf.add, name='b_residual4/add2')
        t2 = ElementwiseLayer([t2, temp_global], combine_fn=tf.add, name='b_residual4/add3')
        t2 = SubpixelConv2d(t2, scale=2, n_out_channel=None, act=tf.nn.selu, name='attention4/pixelshufflerx2/1')

        t2 = Conv2d(t2, 32, (3, 3), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                    name='global_n32/s1')
        t2 = Conv2d(t2, 1, (3, 3), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init, name='t2/out')
        return t2


def add_two_layer(t1, t2):
    t = ElementwiseLayer([t1, t2], combine_fn=tf.add, act=tl.act.hard_tanh, name='add_all1')
    return t


def batch_activ_conv(current, out_features, is_train, idx, idx_, task):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)

    current = BatchNormLayer(current, act=tf.nn.selu, is_train=is_train, gamma_init=g_init,
                             name='t%s/BN%s/%s/2' % (task, idx_, idx))
    current = Conv2d(current, out_features, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                     name='t%s/Cov2d%s/%s/2' % (task, idx_, idx))

    current = BatchNormLayer(current, act=tf.nn.selu, is_train=is_train, gamma_init=g_init,
                             name='t%s/BN%s/%s/4' % (task, idx_, idx))
    current = Conv2d(current, out_features, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                     name='t%s/Cov2d%s/%s/4' % (task, idx_, idx))
    return current


def block(input, layers, growth, is_train, idx_, task):
    input = Conv2d(input, growth, (3, 3), (1, 1), act=tf.nn.selu, padding='SAME', name='t%s/Cov2d%s/input' % (task, idx_))
    current = input
    for idx in range(layers):
        tmp = batch_activ_conv(current, growth, is_train=is_train, idx=idx, idx_=idx_, task=task)
        current = ConcatLayer([current, tmp], 3, name='t%s/concat%s/%s' % (task, idx_, idx))
    return current


def softmax(x):
    return tf.divide(tf.exp(x), tf.expand_dims(tf.expand_dims(tf.expand_dims(
        tf.reduce_sum(tf.exp(x), axis=[1, 2, 3]), -1), -1), -1))


def bottleneck(x, is_train, con_number, block_num):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    x_local_temp = x
    x = BatchNormLayer(x, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='residual%s/b1' % block_num)
    x = Conv2d(x, con_number / 4, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='residual%s/c1' % block_num)
    x = BatchNormLayer(x, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='residual%s/b2' % block_num)
    x = Conv2d(x, con_number / 4, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='residual%s/c2' % block_num)
    x = BatchNormLayer(x, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='residual%s/b3' % block_num)
    x = Conv2d(x, con_number, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='residual%s/c3' % block_num)
    x = ElementwiseLayer([x, x_local_temp], combine_fn=tf.add, name='residual%s/add' % block_num)
    return x


def attention(x, y, con_number, block_num):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    x_temp = x
    x = Conv2d(x, (con_number / 2), (3, 3), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                 name='attention%s/xreduce' % block_num)
    y = Conv2d(y, (con_number / 2), (3, 3), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
               name='attention%s/yreduce' % block_num)
    a_c = x
    a_s = x
    a_c = GlobalMeanPool2d(a_c, name='attention%s/pooling' % block_num)
    a_c = DenseLayer(a_c, con_number / 2, act=tf.nn.selu, name='attention%s/dense1' % block_num)
    a_c = DenseLayer(a_c, con_number / 2, act=tf.nn.sigmoid, name='attention%s/dense2' % block_num)
    a_c = ReshapeLayer(a_c, (-1, 1, 1, int(con_number / 2)), name='attention%s/reshape' % block_num)

    a_s = Conv2d(a_s, con_number / 4, (1, 1), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                 name='attention%s/spatial1' % block_num)
    a_s = Conv2d(a_s, 1, (1, 1), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                 name='attention%s/spatial2' % block_num)
    spa = a_s
    a_c = ElementwiseLayer([y, a_c], combine_fn=tf.multiply, name='attention%s/fusion/mult1' % block_num)
    a_s = ElementwiseLayer([a_c, a_s], combine_fn=tf.multiply, name='attention%s/fusion/mult2' % block_num)
    a_s = ConcatLayer([x, a_s], concat_dim=3, name='attention%s/concat1' % block_num)

    a_s = Conv2d(a_s, con_number, (3, 3), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                name='attention%s/dimreduce' % block_num)
    x = ElementwiseLayer([x_temp, a_s], combine_fn=tf.add, name='attention%s/fusion/add' % block_num)
    return x, spa