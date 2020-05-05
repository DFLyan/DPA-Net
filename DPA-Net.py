#! /usr/bin/python
# -*- coding: utf8 -*-

import time, os, random

import tensorflow as tf
import tensorlayer as tl
import math as ma
from model_DPA_optim import *
from utils import *
from config import config
from skimage.measure import compare_psnr


batch_size = config.TRAIN.batch_size
lr_1 = config.TRAIN.lr_1
lr_2 = config.TRAIN.lr_2
lr_decay = config.TRAIN.lr_decay
beta1 = config.TRAIN.beta1
n_epoch_dpa_1 = config.TRAIN.n_epoch_dpa_1
n_epoch_dpa_2 = config.TRAIN.n_epoch_dpa_2
decay_every_dpa = config.TRAIN.decay_every_dpa
decay_every_dpa2 = config.TRAIN.decay_every_dpa2
ni = int(4)
ni_ = int(batch_size//4)

block_size = config.TRAIN.block_size
MR = config.TRAIN.MR
imagesize = block_size * block_size
size_y = ma.ceil(block_size * block_size * MR)
tv_weight = config.TRAIN.tv_weight
Num = 1
fullimgsize = 64
num_block = int(fullimgsize / block_size)


def read_all_imgs(img_list, path='', n_threads=32):
    """ Returns all images in array by given path and name of each image file. """
    imgs = []
    for idx in range(0, len(img_list), n_threads):
        b_imgs_list = img_list[idx : idx + n_threads]
        b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_gray_imgs_fn, path=path)
        # print(b_img16s.shape)
        imgs.extend(b_imgs)
        print('read %d from %s' % (len(imgs), path))
    return imgs


def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)


def train():
    save_dir_DPA_1 = ("samples/DPA/%s/train/%s_g/1" % (Num, MR)).format(tl.global_flag['mode'])
    save_dir_DPA_2 = ("samples/DPA/%s/train/%s_g/2" % (Num, MR)).format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_DPA_1)
    tl.files.exists_or_mkdir(save_dir_DPA_2)
    checkpoint_dir = "checkpoint/DPA/%s/%s" % (Num, MR)  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)

    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))

    t_target_image = tf.placeholder('float32', [None, None, None, 1], name='t_target_image')
    t_block_image = tf.placeholder('float32', [None, block_size, block_size, 1], name='t_block_image')
    y1_image = tf.placeholder('float32', [None, None, None, size_y], name='y1_image')
    if not os.path.isfile("Gaussian%s_16.npy" % MR):
        A = np.random.normal(loc=0, scale=(1/size_y), size=[imagesize, int(size_y)])
        A = A.astype(np.float32)
        np.save("Gaussian%s_16.npy" % MR, A)
    else:
        A = np.load("Gaussian%s_16.npy" % MR, encoding='latin1')

    x_hat = tf.reshape(t_block_image, [batch_size, imagesize])
    y_meas = tf.matmul(x_hat, A)


    t1, t2_a1, t2_a2, t2_a3, t2_a4 = T1(y1_image, is_train=True, reuse=False)
    t2 = T2(y1_image, t2_a1, t2_a2, t2_a3, t2_a4, is_train=True, reuse=False)
    t = add_two_layer(t1, t2)


    t1_, t2_a1_, t2_a2_, t2_a3_, t2_a4_ = T1(y1_image, is_train=False, reuse=True)
    t2_ = T2(y1_image, t2_a1_, t2_a2_, t2_a3_, t2_a4_, is_train=False, reuse=True)
    t_ = add_two_layer(t1_, t2_)


    img_p = t.outputs
    y_meas_p = tf.zeros([batch_size, 1, num_block, size_y])
    for num_r in range(1, num_block + 1):
        y_temp_p = tf.zeros([batch_size, 1, 1, size_y])
        for num_c in range(1, num_block + 1):
            img_block = img_p[:, (num_r - 1) * block_size:num_r * block_size,
                        (num_c - 1) * block_size:num_c * block_size, :]
            img_block = tf.reshape(img_block, [batch_size, block_size, block_size, 1])
            x_res = tf.reshape(img_block, [batch_size, imagesize])
            y_meas_ = tf.matmul(x_res, A)
            y_meas_ = tf.reshape(y_meas_, [batch_size, 1, 1, size_y])
            if num_c - 1 == 0:
                y_temp_p = y_temp_p + y_meas_
            else:
                y_temp_p = tf.concat([y_temp_p, y_meas_], 2)
        if num_r - 1 == 0:
            y_meas_p = y_meas_p + y_temp_p
        else:
            y_meas_p = tf.concat([y_meas_p, y_temp_p], 1)


    tv_loss_t1 = tv_weight * (
            tf.reduce_mean(tf.reduce_sum(tf.abs(t1.outputs - tf.concat([t1.outputs[:, fullimgsize - 1:fullimgsize, :, :],
                                                         t1.outputs[:, :fullimgsize - 1, :, :]], 1)), [1, 2, 3]))
            +
            tf.reduce_mean(tf.reduce_sum(tf.abs(t1.outputs - tf.concat([t1.outputs[:, :, fullimgsize - 1:fullimgsize, :],
                                                         t1.outputs[:, :, :fullimgsize - 1, :]], 2)), [1, 2, 3]))
                             )


    ade_loss_t1 = tl.cost.absolute_difference_error(t1.outputs, t_target_image, is_mean=True)

    ade_loss_t2 = tl.cost.absolute_difference_error(t.outputs, t_target_image, is_mean=True)

    meas_loss = tl.cost.absolute_difference_error(y1_image, y_meas_p, is_mean=True)

    mse_loss = tl.cost.mean_squared_error(t.outputs, t_target_image, is_mean=True)

    g_loss_t1 = ade_loss_t1 + tv_loss_t1
    g_loss_t2 = ade_loss_t2 + 1 * meas_loss
    g_loss_t3 = ade_loss_t1 + tv_loss_t1 + ade_loss_t2 + 1*meas_loss

    psnr = tf.constant(10, dtype=tf.float32) * log10(tf.constant(4, dtype=tf.float32) / (mse_loss))
    tf.summary.scalar('psnr', psnr)
    tf.summary.scalar('tv_loss_t1', tv_loss_t1)
    tf.summary.scalar('meas_loss', meas_loss)
    tf.summary.scalar('mse_loss', mse_loss)
    tf.summary.scalar('ade_loss_t1', ade_loss_t1)
    tf.summary.scalar('g_loss_t2', g_loss_t2)



    t1_vars = tl.layers.get_variables_with_name('t1', True, True)
    t2_vars = tl.layers.get_variables_with_name('t2', True, True)

    with tf.variable_scope('learning_rate_dpa'):
            lr_v_1 = tf.Variable(lr_1, trainable=False)
    with tf.variable_scope('learning_rate_dpa'):
            lr_v_2 = tf.Variable(lr_2, trainable=False)

    t1_optim = tf.train.AdamOptimizer(lr_v_1).minimize(g_loss_t1, var_list=t1_vars)
    t2_optim = tf.train.AdamOptimizer(lr_v_1).minimize(g_loss_t2, var_list=t2_vars)
    t3_optim = tf.train.AdamOptimizer(lr_v_2).minimize(g_loss_t3, var_list=[t1_vars, t2_vars])


    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(
        config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options))


    # merged = tf.summary.merge_all()
    # writer_2 = tf.summary.FileWriter("logs/DPA/%s/c%s/2" % (Num, MR), tf.get_default_graph())

    sess.run(tf.global_variables_initializer())
    if tl.files.load_and_assign_npz(
            sess=sess, name=checkpoint_dir+'/g_{}_2.npz'.format(tl.global_flag['mode']), network=t) is False:
        tl.files.load_and_assign_npz(
            sess=sess, name=checkpoint_dir + '/g_{}_1.npz'.format(tl.global_flag['mode']), network=t1)



###============================= TRAINING ===============================###
    sample_imgs = read_all_imgs(valid_hr_img_list[0:batch_size], path=config.VALID.hr_img_path, n_threads=16)
    sample_imgs_256 = tl.prepro.threading_data(sample_imgs, fn=norm)
    sample_imgs_256__ = np.reshape(sample_imgs_256.astype(np.float32), [1, 128, 128, 1])

    size = sample_imgs_256.shape

    a = int(ma.ceil(size[1] / block_size))

    global y_fullimg_sample
    y_fullimg_sample = np.zeros((1, a, a, size_y))
    for num_r in range(1, int(ma.ceil(size[1] / block_size)) + 1):
        for num_c in range(1, int(ma.ceil(size[2] / block_size)) + 1):
            img_block = sample_imgs_256[:, (num_r - 1) * block_size:num_r * block_size,
                        (num_c - 1) * block_size:num_c * block_size]
            img_block = np.reshape(img_block.astype(np.float32), [1, block_size, block_size, 1])
            x_hat = np.reshape(img_block, [1, imagesize])
            y_meas_ = np.matmul(x_hat, A)
            y_meas_ = np.reshape(y_meas_, [1, 1, 1, size_y])
            y_fullimg_sample[:, (num_r - 1):num_r, (num_c - 1):num_c, :] = y_meas_

    print('sample HR sub-image:',sample_imgs_256.shape, sample_imgs_256.min(), sample_imgs_256.max())
    tl.vis.save_images(sample_imgs_256__, [1, 1], save_dir_DPA_1 + '/_train_sample.png')
    tl.vis.save_images(sample_imgs_256__, [1, 1], save_dir_DPA_2 + '/_train_sample.png')

    ##step1###
    for epoch in range(0, n_epoch_dpa_1):
        if epoch != 0 and (epoch % decay_every_dpa == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every_dpa)
            sess.run(tf.assign(lr_v_1, lr_1 * new_lr_decay))
            log = " ** new learning rate: %f " % (lr_1 * new_lr_decay)
            print(log)
        elif epoch == 0:
            sess.run(tf.assign(lr_v_1, lr_1))
            log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f " % (lr_1, decay_every_dpa, lr_decay)
            print(log)

        epoch_time = time.time()
        total_t1_loss, total_t2_loss, total_mse_loss, n_iter_DPA = 0, 0, 0, 0

        if epoch == 0:
            global sum1
            sum1 = 0
        else:
            pass

        random.shuffle(train_hr_img_list)
        for idx in range(0, int(len(train_hr_img_list) // batch_size)):
            step_time = time.time()
            b_imgs_list = train_hr_img_list[idx * batch_size: (idx + 1) * batch_size]
            b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_gray_imgs_fn, path=config.TRAIN.hr_img_path)
            b_imgs = tl.prepro.threading_data(b_imgs, fn=norm)
            b_imgs = tl.prepro.threading_data(b_imgs, fn=augm)

            size = b_imgs.shape

            a = int(ma.ceil(size[1] / block_size))

            global y_fullimg
            y_fullimg = np.zeros((batch_size, a, a, size_y))
            for num_r in range(1, int(ma.ceil(size[1] / block_size)) + 1):
                for num_c in range(1, int(ma.ceil(size[2] / block_size)) + 1):
                    img_block = b_imgs[:, (num_r - 1) * block_size:num_r * block_size,
                                (num_c - 1) * block_size:num_c * block_size]
                    img_block = np.reshape(img_block, [batch_size, block_size, block_size, 1])
                    y_meas_ = sess.run(y_meas, feed_dict={t_block_image: img_block})
                    y_meas_ = np.reshape(y_meas_, [batch_size, 1, 1, size_y])
                    y_fullimg[:, (num_r - 1):num_r, (num_c - 1):num_c, :] = y_meas_

            b_imgs = np.reshape(b_imgs, [batch_size, size[1], size[2], 1])

            errt1, errt1_1, errt1_2, __ = sess.run(
                [g_loss_t1, ade_loss_t1, tv_loss_t1, t1_optim],
                {y1_image: y_fullimg, t_target_image: b_imgs})

            errt2, errt2_1, errt2_2, ___ = sess.run(
                [g_loss_t2, ade_loss_t2, meas_loss, t2_optim],
                {y1_image: y_fullimg, t_target_image: b_imgs})

            if n_iter_DPA % 500 == 0:
                print(
                    "Epoch [%2d/%2d] %4d time:%4.4fs,t1_loss: %.4f(t1_ade:%.4f, tv_loss:%.4f),"
                    "t2_loss: %.4f(t2_ade: %.4f, meas: %.4f)" % (
                        epoch, n_epoch_dpa_1, n_iter_DPA, time.time() - step_time, errt1, errt1_1, errt1_2,
                        errt2, errt2_1, errt2_2,))

            # if sum1 % 50 == 0:
            #     writer_1.add_summary(summary, sum1)

            total_t1_loss += errt1
            total_t2_loss += errt2
            n_iter_DPA += 1
            sum1 += 1

        log = "[*] Epoch1: [%2d/%2d] time: %4.4fs, t1_loss: %.8f, t2_loss:%.8f" % (
            epoch, n_epoch_dpa_1, time.time() - epoch_time, total_t1_loss / n_iter_DPA, total_t2_loss / n_iter_DPA)
        print(log)

        ## quick evaluation on train set
        if epoch % 1 == 0:
            out_T1, out_T2, out_T = sess.run([t1_.outputs, t2_.outputs, t_.outputs], {
                y1_image: y_fullimg_sample})  # ; print('gen sub-image:', out.shape, out.min(), out.max())
            print("[*] save images")
            out_T1 = np.reshape(out_T1, [1, 128, 128, 1])
            out_T1_inv = tl.prepro.threading_data(out_T1, fn=inv_norm)
            out_T2 = np.reshape(out_T2, [1, 128, 128, 1])
            out_T2_inv = tl.prepro.threading_data(out_T2, fn=inv_norm)
            out_DPA = np.reshape(out_T, [1, 128, 128, 1])
            out_DPA_inv = tl.prepro.threading_data(out_DPA, fn=inv_norm)
            tl.vis.save_images(out_T1_inv.astype(np.uint8), [1, 1], save_dir_DPA_2 + '/train_%d_T1.png' % epoch)
            tl.vis.save_images(out_T2_inv.astype(np.uint8), [1, 1], save_dir_DPA_2 + '/train_%d_T2.png' % epoch)
            tl.vis.save_images(out_DPA_inv.astype(np.uint8), [1, 1], save_dir_DPA_2 + '/train_%d_T.png' % epoch)
            psnr = compare_psnr(out_DPA, sample_imgs_256.reshape([1, 128, 128, 1]))
            print("valid_psnr:%.4f" % psnr)

        ## save model
        if (epoch != 0) and (epoch % 1 == 0):
            tl.files.save_npz(t.all_params, name=checkpoint_dir + '/g_{}_1.npz'.format(tl.global_flag['mode']),
                              sess=sess)


    for epoch in range(0,n_epoch_dpa_2):
        if epoch !=0 and (epoch % decay_every_dpa2 == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every_dpa2)
            sess.run(tf.assign(lr_v_2, lr_2 * new_lr_decay))
            log = " ** new learning rate: %f (for GAN)" % (lr_2 * new_lr_decay)
            print(log)
        elif epoch == 0:
            sess.run(tf.assign(lr_v_2, lr_2))
            log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (lr_2, decay_every_dpa2, lr_decay)
            print(log)

        epoch_time = time.time()
        total_t1_loss, total_t2_loss, total_mse_loss, n_iter_DPA = 0, 0, 0, 0

        if epoch == 0:
            global sum3
            sum3 = 0
        else:
            pass

        random.shuffle(train_hr_img_list)
        for idx in range(0, int(len(train_hr_img_list)//batch_size)):
            step_time = time.time()
            b_imgs_list = train_hr_img_list[idx * batch_size: (idx + 1) * batch_size]
            b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_gray_imgs_fn, path=config.TRAIN.hr_img_path)
            b_imgs = tl.prepro.threading_data(b_imgs, fn=norm)
            b_imgs = tl.prepro.threading_data(b_imgs, fn=augm)

            size = b_imgs.shape

            a = int(ma.ceil(size[1] / block_size))

            y_fullimg = np.zeros((batch_size, a, a, size_y))
            for num_r in range(1, int(ma.ceil(size[1] / block_size)) + 1):
                for num_c in range(1, int(ma.ceil(size[2] / block_size)) + 1):
                    img_block = b_imgs[:, (num_r - 1) * block_size:num_r * block_size,
                                (num_c - 1) * block_size:num_c * block_size]
                    img_block = np.reshape(img_block, [batch_size, block_size, block_size, 1])
                    y_meas_ = sess.run(y_meas, feed_dict={t_block_image: img_block})
                    y_meas_ = np.reshape(y_meas_, [batch_size, 1, 1, size_y])
                    y_fullimg[:, (num_r - 1):num_r, (num_c - 1):num_c, :] = y_meas_

            b_imgs = np.reshape(b_imgs, [batch_size, size[1], size[2], 1])

            errt2, errt2_1, errt2_2, errM, errTV, ___ = sess.run(
                [g_loss_t3, ade_loss_t2, meas_loss, mse_loss, tv_loss_t1, t3_optim],
                {y1_image: y_fullimg, t_target_image: b_imgs})

            if n_iter_DPA % 500 == 0:
                print(
                    "Epoch2 [%2d/%2d] %4d time:%4.4fs,t2_loss: %.4f(t2_ade: %.4f, meas: %.4f),mse_loss:%.4f,tv_loss:%.4f" % (
                    epoch, n_epoch_dpa_2, n_iter_DPA, time.time() - step_time, errt2, errt2_1, errt2_2, errM, errTV))

            # if sum2 % 50 == 0:
            #     writer_2.add_summary(summary, sum2)

            total_t2_loss += errt2
            total_mse_loss += errM
            n_iter_DPA += 1
            sum3 +=1

        log = "[*] Epoch2: [%2d/%2d] time: %4.4fs, t1_loss: %.8f, t2_loss: %.8f, mse_loss:%.8f" % (
        epoch, n_epoch_dpa_2, time.time() - epoch_time, total_t1_loss / n_iter_DPA, total_t2_loss / n_iter_DPA, total_mse_loss / n_iter_DPA)
        print(log)


        ## quick evaluation on train set
        if epoch % 1 == 0:
            out_T1, out_T2, out_T = sess.run([t1_.outputs, t2_.outputs, t_.outputs], {y1_image: y_fullimg_sample})  # ; print('gen sub-image:', out.shape, out.min(), out.max())
            print("[*] save images")
            out_T1 = np.reshape(out_T1, [1, 128, 128, 1])
            out_T1_inv = tl.prepro.threading_data(out_T1, fn=inv_norm)
            out_T2 = np.reshape(out_T2, [1, 128, 128, 1])
            out_T2_inv = tl.prepro.threading_data(out_T2, fn=inv_norm)
            out_DPA = np.reshape(out_T, [1, 128, 128, 1])
            out_DPA_inv = tl.prepro.threading_data(out_DPA, fn=inv_norm)
            tl.vis.save_images(out_T1_inv.astype(np.uint8), [1, 1], save_dir_DPA_2 + '/train_%d_T1.png' % epoch)
            tl.vis.save_images(out_T2_inv.astype(np.uint8), [1, 1], save_dir_DPA_2 + '/train_%d_T2.png' % epoch)
            tl.vis.save_images(out_DPA_inv.astype(np.uint8), [1, 1], save_dir_DPA_2 + '/train_%d_T.png' % epoch)
            psnr = compare_psnr(out_DPA, sample_imgs_256.reshape([1, 128, 128, 1]))
            print("valid_psnr:%.4f" % psnr)


        ## save model
        if (epoch != 0) and (epoch % 1 == 0):
            tl.files.save_npz(t.all_params, name=checkpoint_dir + '/g_{}_2.npz'.format(tl.global_flag['mode']), sess=sess)




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='GRAY', help='GRAY, evaluate')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'GRAY':
        train()
    else:
        raise Exception("Unknow --mode")
