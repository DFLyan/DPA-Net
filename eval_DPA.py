#! /usr/bin/python
# -*- coding: utf8 -*-

import os, time
import tensorflow as tf
import tensorlayer as tl
import math as ma
from model_DPA import *
from utils import *
from config import config
from skimage.measure import compare_psnr

block_size = config.TRAIN.block_size
MR = config.TRAIN.MR
imagesize = block_size * block_size
size_y = ma.ceil(block_size * block_size * MR)
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



def evaluate():
    save_dir = ("samples/DPA/%s/test/%s_g" % (Num, MR)).format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "checkpoint/DPA/%s/%s" % (Num, MR)


    ###====================== PRE-LOAD DATA ===========================###
    test_hr_img_list = sorted(tl.files.load_file_list(path=config.TEST.hr_img_path, regx='.*.*', printable=False))
    test_hr_imgs = read_all_imgs(test_hr_img_list, path=config.TEST.hr_img_path, n_threads=32)

    ###========================== DEFINE MODEL ============================###
    t_block_image = tf.placeholder('float32', [None, block_size, block_size, 1], name='t_block_image')
    y1_image = tf.placeholder('float32', [None, None, None, size_y], name='y1_image')


    A = np.load("Gaussian%s_16.npy" % MR, encoding='latin1')

    x_hat = tf.reshape(t_block_image, [1, imagesize])
    y_meas = tf.matmul(x_hat, A)

    t1_, t2_a1_, t2_a2_, t2_a3_, t2_a4_ = T1(y1_image, is_train=False, reuse=False)
    t2_, spa1_, spa2_, spa3_, spa4_ = T2(y1_image, t2_a1_, t2_a2_, t2_a3_, t2_a4_, is_train=False, reuse=False)
    t_ = add_two_layer(t1_, t2_)


    ###========================== RESTORE G =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/g_GRAY_2.npz', network=t_)

    ###======================= EVALUATION =============================###
    global sum, sum_t
    sum = 0
    sum_t = 0
    f1 = open("samples/DPA/%s.txt" % MR, "w")
    for imid in range(0, len(test_hr_imgs)):
        b_imgs_ = tl.prepro.threading_data(test_hr_imgs[imid:imid+1], fn=norm)
        size = b_imgs_.shape

        b_imgs_ = np.reshape(b_imgs_, [size[1], size[2]])

        a = int(ma.ceil(size[1] / block_size))

        # count = 0
        global img

        global y_full
        y_full = np.zeros((1, a, a, size_y))
        for num_r in range(1, int(ma.ceil(size[1] / block_size)) + 1):
            for num_c in range(1, int(ma.ceil(size[2] / block_size)) + 1):
                img_block = b_imgs_[(num_r - 1) * block_size:num_r * block_size,
                            (num_c - 1) * block_size:num_c * block_size]
                img_block = np.reshape(img_block, [-1, block_size, block_size, 1])
                y_meas_ = sess.run(y_meas, feed_dict={t_block_image: img_block})
                y_meas_ = np.reshape(y_meas_, [-1, 1, 1, size_y])
                noise = np.random.normal(loc=0, scale=0, size=[1, 1, 1, size_y])
                y_meas_ = y_meas_ + noise
                y_full[:, (num_r - 1):num_r, (num_c - 1):num_c, :] = y_meas_

        y_fullimg = y_full

        start_time = time.time()
        img1, img2, img, spa1, spa2, spa3, spa4 = sess.run([t1_.outputs, t2_.outputs, t_.outputs, spa1_.outputs, spa2_.outputs, spa3_.outputs, spa4_.outputs],
                                                           feed_dict={y1_image: y_fullimg})
        print("took: %4.4fs" % (time.time() - start_time))
        sum_t += (time.time() - start_time)

        img = np.reshape(img, [size[1], size[2]])
        img1 = np.reshape(img1, [size[1], size[2]])
        img2 = np.reshape(img2, [size[1], size[2]])
        spa1 = np.squeeze(spa1)

        img = img[:size[1], :size[2]]

        psnr = compare_psnr(b_imgs_.astype(np.float32), img)
        print("%s's PSNR:%.8f" % (test_hr_img_list[imid], psnr))
        f1.write("%.8f" % psnr)
        f1.write("\n")

        sum += psnr


        print("[*] save images")
        img1 = tl.prepro.threading_data(img1[:size[1], :size[2]], fn=inv_norm)
        save_image(img1.astype(np.uint8), save_dir+'/%s_gen_t1.png' % test_hr_img_list[imid])
        img2 = tl.prepro.threading_data(img2[:size[1], :size[2]], fn=inv_norm)
        save_image(img2.astype(np.uint8), save_dir+'/%s_gen_t2.png' % test_hr_img_list[imid])
        img = tl.prepro.threading_data(img, fn=inv_norm)
        save_image(img.astype(np.uint8), save_dir+'/%s_gen.png' % test_hr_img_list[imid])
        save_image(spa1, save_dir + '/%s_gen_spa1.png' % test_hr_img_list[imid])
        b_imgs_ = tl.prepro.threading_data(b_imgs_, fn=inv_norm)
        save_image(b_imgs_.astype(np.uint8), save_dir+'/%s_hr.png' % test_hr_img_list[imid])


    f1.close()
    psnr_a = sum / len(test_hr_imgs)
    print("PSNR_AVERAGE:%.8f" % psnr_a)
    time_a = sum_t / len(test_hr_imgs)
    print("TIME_AVERAGE:%.8f" % time_a)


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


