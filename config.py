from easydict import EasyDict as edict
import json
import math as ma

config = edict()
config.TRAIN = edict()
config.VALID = edict()
config.TEST = edict()

config.TRAIN.block_size = 16
config.TRAIN.MR = 0.5
config.TRAIN.tv_weight = 0.00018

config.TRAIN.batch_size = 16

config.TRAIN.lr_1 = 0.0001
config.TRAIN.lr_2 = 0.0001

config.TRAIN.beta1 = 0.9

config.TRAIN.n_epoch_dpa_1 = 30
config.TRAIN.n_epoch_dpa_2 = 360
config.TRAIN.lr_decay = 0.8

config.TRAIN.decay_every_dpa = int(30)
config.TRAIN.decay_every_dpa2 = int(30)

config.TRAIN.hr_img_path = 'data/train/'

config.VALID.hr_img_path = 'data/valid/'

config.TEST.hr_img_path = 'data/SET11/'
# config.TEST.hr_img_path = 'data/SET5_gray/'
# config.TEST.hr_img_path = 'data/BSD68/'

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
