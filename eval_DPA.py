#! /usr/bin/python
# -*- coding: utf8 -*-

import os, time, logging
import tensorflow as tf
import tensorlayer as tl
import math as ma
from model_DPA_optim import *
from utils import *
from config import config
from skimage.measure import compare_psnr, compare_ssim
logging.getLogger().setLevel(logging.INFO)

block_size = config.TRAIN.block_size
MR = config.TRAIN.MR
imagesize = block_size * block_size
size_y = ma.ceil(block_size * block_size * MR)
Num = 1
nsv = 0
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



def evaluate():
    save_dir = ("samples/DPA/%s/test/%s_g/noise_standard_variance_%s" % (Num, MR, nsv)).format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "checkpoint/DPA/%s/%s" % (Num, MR)


    ###====================== PRE-LOAD DATA ===========================###
    test_hr_img_list = sorted(tl.files.load_file_list(path=config.TEST.hr_img_path, regx='.*.*', printable=True))
    test_hr_imgs = read_all_imgs(test_hr_img_list, path=config.TEST.hr_img_path, n_threads=32)

    ###========================== DEFINE MODEL ============================###
    y1_image = tf.placeholder('float32', [None, None, None, size_y], name='y1_image')

    A = np.load("Gaussian%s_16.npy" % MR, encoding='latin1')

    t1_, t2_a1_, t2_a2_, t2_a3_, t2_a4_ = T1(y1_image, is_train=False, reuse=False)
    t2_ = T2(y1_image, t2_a1_, t2_a2_, t2_a3_, t2_a4_, is_train=False, reuse=False)
    t_ = add_two_layer(t1_, t2_)

    ###========================== RESTORE G =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    t_.print_layers()
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/g_GRAY_2.npz', network=t_)

    ###======================= EVALUATION =============================###
    global sum, sum_t, sum_s
    sum = 0
    sum_t = 0
    sum_s = 0
    if config.TEST.hr_img_path == 'data/SET11/':
        dataset = 'SET11'
    else:
        dataset = 'BSD68'
    f1 = open("samples/DPA/%s/%s_%s.txt" % (Num, dataset, MR), "w")
    for imid in range(0, len(test_hr_imgs)):
        b_imgs_ = tl.prepro.threading_data(test_hr_imgs[imid:imid+1], fn=norm)
        size = b_imgs_.shape
        b_imgs_ = np.reshape(b_imgs_, [size[1], size[2]])

        row_pad = block_size - size[1]%block_size
        col_pad = block_size - size[2]%block_size

        if (size[1] % block_size == 0) & (size[2] % block_size == 0):
            im_pad = b_imgs_
        else:
            im_pad = np.concatenate((b_imgs_, np.zeros((size[1], col_pad))), axis=1)
            im_pad = np.concatenate((im_pad, np.zeros((row_pad, size[2]+col_pad))), axis=0)


        size_p = im_pad.shape
        global img
        a = int(ma.ceil(size_p[0] / block_size))
        b = int(ma.ceil(size_p[1] / block_size))
        noise_img = im_pad

        good_img = noise_img
        bad_img = good_img

        X = np.split(bad_img, a, 0)
        Y = []
        for x in X:
            X_ = np.split(x, b, 1)
            Y_ = []
            for x_ in X_:
                x_ = np.reshape(x_, [size[0], imagesize])
                y_meas_ = np.matmul(x_, A)
                y_meas_ = np.reshape(y_meas_, [size[0], 1, 1, size_y])
                noise = np.random.normal(loc=0, scale=nsv, size=[1, 1, 1, size_y])
                y_meas_ = y_meas_ + noise
                Y_.append(y_meas_)
            y_meas_c = np.concatenate([y_ for y_ in Y_], 1)
            Y.append(y_meas_c)
        y_full = np.concatenate([y for y in Y], 0)

        y_fullimg = np.reshape(y_full, [1, a, b, size_y])

        start_time = time.time()
        img1, img2, img = sess.run([t1_.outputs, t2_.outputs, t_.outputs],
                                                           feed_dict={y1_image: y_fullimg})
        print("took: %4.4fs" % (time.time() - start_time))
        sum_t += (time.time() - start_time)

        img = np.reshape(img, [size_p[0], size_p[1]])
        img1 = np.reshape(img1, [size_p[0], size_p[1]])
        img2 = np.reshape(img2, [size_p[0], size_p[1]])
        # spa1 = np.squeeze(spa1)

        img = img[:size[1], :size[2]]

        psnr = compare_psnr((b_imgs_.astype(np.float32)+1)*127.5, (img+1)*127.5, data_range=255)
        print("%s's PSNR:%.8f" % (test_hr_img_list[imid], psnr))
        ssim = compare_ssim(X=(b_imgs_.astype(np.float32)+1)*127.5, Y=(img.astype(np.float32)+1)*127.5, multichannel=False, data_range=255)
        print("%s's SSIM:%.8f" % (test_hr_img_list[imid], ssim))
        f1.write("%.8f,%.8f" % (psnr,ssim))
        f1.write("\n")

        sum += psnr
        sum_s += ssim


        print("[*] save images")
        img1 = tl.prepro.threading_data(img1[:size[1], :size[2]], fn=inv_norm)
        save_image(img1.astype(np.uint8), save_dir+'/%s_gen_t1.png' % test_hr_img_list[imid])
        img2 = tl.prepro.threading_data(img2[:size[1], :size[2]], fn=inv_norm)
        save_image(img2.astype(np.uint8), save_dir+'/%s_gen_t2.png' % test_hr_img_list[imid])
        img = tl.prepro.threading_data(img, fn=inv_norm)
        save_image(img.astype(np.uint8), save_dir+'/%s_gen_psnr%.4f_ssim%.4f.png' % (test_hr_img_list[imid], psnr, ssim))
        # spa1 = tl.prepro.threading_data(spa1, fn=inv_norm_0)
        # save_image(spa1.astype(np.uint8), save_dir + '/%s_gen_spa4.png' % test_hr_img_list[imid])
        b_imgs_ = tl.prepro.threading_data(b_imgs_, fn=inv_norm)
        save_image(b_imgs_.astype(np.uint8), save_dir+'/%s_hr.png' % test_hr_img_list[imid])


    f1.close()
    psnr_a = sum / len(test_hr_imgs)
    print("PSNR_AVERAGE:%.8f" % psnr_a)
    time_a = sum_t / len(test_hr_imgs)
    print("TIME_AVERAGE:%.8f" % time_a)
    ssim_a = sum_s / len(test_hr_imgs)
    print("SSIM_AVERAGE:%.8f" % ssim_a)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='evaluate_gray', help='evaluate_gray')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'evaluate_gray':
        evaluate()
    else:
        raise Exception("Unknow --mode")


