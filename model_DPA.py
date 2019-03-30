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

        t1_local_temp = t1
        t1 = BatchNormLayer(t1, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='residual1/b1')
        t1 = Conv2d(t1, 64, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='residual1/c1')
        t1 = BatchNormLayer(t1, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='residual1/b2')
        t1 = Conv2d(t1, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='residual1/c2')
        t1 = BatchNormLayer(t1, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='residual1/b3')
        t1 = Conv2d(t1, 256, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='residual1/c3')
        t1 = ElementwiseLayer([t1, t1_local_temp], combine_fn=tf.add, name='residual1/add')
        t1_a1 = t1
        t1 = SubpixelConv2d(t1, scale=2, n_out_channel=None, act=None, name='pixelshufflerx2/1')
        t1 = Conv2d(t1, 256, (3, 3), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                    name='b_residual1/n16/2')

        t1_local_temp = t1
        t1 = BatchNormLayer(t1, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='residual2/b1')
        t1 = Conv2d(t1, 64, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                    name='residual2/c1')
        t1 = BatchNormLayer(t1, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='residual2/b2')
        t1 = Conv2d(t1, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                    name='residual2/c2')
        t1 = BatchNormLayer(t1, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='residual2/b3')
        t1 = Conv2d(t1, 256, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                    name='residual2/c3')
        t1 = ElementwiseLayer([t1, t1_local_temp], combine_fn=tf.add, name='residual2/add')
        t1_a2 = t1
        t1 = SubpixelConv2d(t1, scale=2, n_out_channel=None, act=None, name='pixelshufflerx2/2')
        t1 = Conv2d(t1, 256, (3, 3), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                    name='b_residual2/n16/2')

        t1_local_temp = t1
        t1 = BatchNormLayer(t1, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='residual3/b1')
        t1 = Conv2d(t1, 64, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                    name='residual3/c1')
        t1 = BatchNormLayer(t1, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='residual3/b2')
        t1 = Conv2d(t1, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                    name='residual3/c2')
        t1 = BatchNormLayer(t1, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='residual3/b3')
        t1 = Conv2d(t1, 256, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                    name='residual3/c3')
        t1 = ElementwiseLayer([t1, t1_local_temp], combine_fn=tf.add, name='residual3/add')
        t1_a3 = t1
        t1 = SubpixelConv2d(t1, scale=2, n_out_channel=None, act=None, name='pixelshufflerx2/3')
        t1 = Conv2d(t1, 256, (3, 3), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                    name='b_residual3/n16/2')

        t1_local_temp = t1
        t1 = BatchNormLayer(t1, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='residual4/b1')
        t1 = Conv2d(t1, 64, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                    name='residual4/c1')
        t1 = BatchNormLayer(t1, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='residual4/b2')
        t1 = Conv2d(t1, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                    name='residual4/c2')
        t1 = BatchNormLayer(t1, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='residual4/b3')
        t1 = Conv2d(t1, 256, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                    name='residual4/c3')
        t1 = ElementwiseLayer([t1, t1_local_temp], combine_fn=tf.add, name='residual4/add')
        t1_a4 = t1
        t1 = SubpixelConv2d(t1, scale=2, n_out_channel=None, act=None, name='pixelshufflerx2/4')

        t1 = Conv2d(t1, 32, (3, 3), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                    name='global_n32/s1')

        t1 = Conv2d(t1, 1, (3, 3), (1, 1), act=tl.act.hard_tanh, padding='SAME', W_init=w_init, b_init=b_init, name='out')
        return t1, t1_a1, t1_a2, t1_a3, t1_a4


def T2(y_2, t2_a1, t2_a2, t2_a3, t2_a4, is_train=False, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = tf.constant_initializer(value=0.0)

    j = 1
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
        t2 = ConcatLayer([t2, t2_a1], concat_dim=3, name='attention1/concat1')
        a_c = t2
        a_s = t2
        a_c = GlobalMeanPool2d(a_c, name='attention1/pooling')
        a_c = DenseLayer(a_c, 512, act=tf.nn.selu, name='attention1/dense1')
        a_c = DenseLayer(a_c, 512, act=tf.nn.softmax, name='attention1/dense2')
        a_c = ReshapeLayer(a_c, (-1, 1, 1, 512), name='attention1/reshape')
        a_s = Conv2d(a_s, 128, (3, 3), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                     name='attention1/spatial1')
        a_s = Conv2d(a_s, 1, (1, 1), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                     name='attention1/spatial2')
        spa1 = a_s
        a_c = ElementwiseLayer([t2, a_c], combine_fn=tf.multiply, name='attention1/fusion/mult1')
        a_s = ElementwiseLayer([a_c, a_s], combine_fn=tf.multiply, name='attention1/fusion/mult2')
        t2 = ElementwiseLayer([t2, a_s], combine_fn=tf.add, name='attention1/fusion/add')
        t2 = Conv2d(t2, 256, (1, 1), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                    name='attention1/dimreduce')
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
        t2 = ConcatLayer([t2, t2_a2], concat_dim=3, name='attention2/concat1')
        a_c = t2
        a_s = t2
        a_c = GlobalMeanPool2d(a_c, name='attention2/pooling')
        a_c = DenseLayer(a_c, 512, act=tf.nn.selu, name='attention2/dense1')
        a_c = DenseLayer(a_c, 512, act=tf.nn.softmax, name='attention2/dense2')
        a_c = ReshapeLayer(a_c, (-1, 1, 1, 512), name='attention2/reshape')
        a_s = Conv2d(a_s, 128, (3, 3), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                     name='attention2/spatial1')
        a_s = Conv2d(a_s, 1, (1, 1), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                     name='attention2/spatial2')
        spa2 = a_s
        a_c = ElementwiseLayer([t2, a_c], combine_fn=tf.multiply, name='attention2/fusion/mult1')
        a_s = ElementwiseLayer([a_c, a_s], combine_fn=tf.multiply, name='attention2/fusion/mult2')
        t2 = ElementwiseLayer([t2, a_s], combine_fn=tf.add, name='attention2/fusion/add')
        t2 = Conv2d(t2, 256, (1, 1), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                    name='attention2/dimreduce')
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
        t2 = ConcatLayer([t2, t2_a3], concat_dim=3, name='attention3/concat1')
        a_c = t2
        a_s = t2
        a_c = GlobalMeanPool2d(a_c, name='attention3/pooling')
        a_c = DenseLayer(a_c, 512, act=tf.nn.selu, name='attention3/dense1')
        a_c = DenseLayer(a_c, 512, act=tf.nn.softmax, name='attention3/dense2')
        a_c = ReshapeLayer(a_c, (-1, 1, 1, 512), name='attention3/reshape')
        a_s = Conv2d(a_s, 128, (3, 3), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                     name='attention3/spatial1')
        a_s = Conv2d(a_s, 1, (1, 1), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                     name='attention3/spatial2')
        spa3 = a_s
        a_c = ElementwiseLayer([t2, a_c], combine_fn=tf.multiply, name='attention3/fusion/mult1')
        a_s = ElementwiseLayer([a_c, a_s], combine_fn=tf.multiply, name='attention3/fusion/mult2')
        t2 = ElementwiseLayer([t2, a_s], combine_fn=tf.add, name='attention3/fusion/add')
        t2 = Conv2d(t2, 256, (1, 1), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                    name='attention3/dimreduce')
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
        t2 = ConcatLayer([t2, t2_a4], concat_dim=3, name='attention4/concat1')
        a_c = t2
        a_s = t2
        a_c = GlobalMeanPool2d(a_c, name='attention4/pooling')
        a_c = DenseLayer(a_c, 512, act=tf.nn.selu, name='attention4/dense1')
        a_c = DenseLayer(a_c, 512, act=tf.nn.softmax, name='attention4/dense2')
        a_c = ReshapeLayer(a_c, (-1, 1, 1, 512), name='attention4/reshape')
        a_s = Conv2d(a_s, 128, (3, 3), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                     name='attention4/spatial1')
        a_s = Conv2d(a_s, 1, (1, 1), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                     name='attention4/spatial2')
        spa4 = a_s
        a_c = ElementwiseLayer([t2, a_c], combine_fn=tf.multiply, name='attention4/fusion/mult1')
        a_s = ElementwiseLayer([a_c, a_s], combine_fn=tf.multiply, name='attention4/fusion/mult2')
        t2 = ElementwiseLayer([t2, a_s], combine_fn=tf.add, name='attention4/fusion/add')
        t2 = Conv2d(t2, 256, (1, 1), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, b_init=b_init,
                    name='attention4/dimreduce')
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
        return t2, spa1, spa2, spa3, spa4


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
                             name='t%s/BN%s/%s/3' % (task, idx_, idx))
    current = Conv2d(current, out_features, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                     name='t%s/Cov2d%s/%s/3' % (task, idx_, idx))
    return current


def block(input, layers, growth, is_train, idx_, task):
    input = Conv2d(input, growth, (1, 1), (1, 1), act=None, padding='SAME', name='t%s/Cov2d%s/input' % (task, idx_))
    current = input
    for idx in range(layers):
        tmp = batch_activ_conv(current, growth, is_train=is_train, idx=idx, idx_=idx_, task=task)
        current = ConcatLayer([current, tmp], 3, name='t%s/concat%s/%s' % (task, idx_, idx))
    return current