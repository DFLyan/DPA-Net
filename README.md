# DPA-Net:Dual-path attention network for compressed sensing image reconstruction
```
@article{sun2020dual,
  title={Dual-path attention network for compressed sensing image reconstruction},
  author={Sun, Yubao and Chen, Jiwei and Liu, Qingshan and Liu, Bo and Guo, Guodong},
  journal={IEEE Transactions on Image Processing},
  volume={29},
  pages={9482--9495},
  year={2020},
  publisher={IEEE}
}
```
## Dataset
You can use crop.m to generate training data.

## Use trained model
You can download the trained model from [Google Drive](https://drive.google.com/open?id=1-fvKrbUg7Q0wWhiwpUpXbsBf4SQl50c3). Then put them in the file folder--checkpoint.

"g_GRAY_1.npz" is only the model of structure-path in the network after first step training. If you want to test the whole network including dual-path, "g_GRAY_2.npz" should be in the checkpoint folder.

## Dependencise
* Python 3.6
* Tensorflow 1.10.0
* Tensorlayer 1.11.1
* numpy
* skiamge
## Training
```
$ python DPA-Net.py
```
## Testing
```
$ puthon eval_DPA.py
```
## Others
### First:
You can change the setting in config.py, such as MR(measurement ratio) and so on. The files of trained model parameter and train data will be uploaded soon.
### Second:
Maybe, some warnings or errors will arise when you run the code because of my careless. If you can not cover it, just give me feedback. I will solve it as soon as possible.

# New
## If you want to simplify the generation of y from the image, you can modify the code:
### Firstly, reshape the measurement matrix:
```
phi = np.reshape(A, (block_size, block_size, 1, size_y))
```
### Secondly, use the function "tf.nn.conv2d" to realize the samling(measurement), for example:
```
y_meas = tf.nn.conv2d(t_target_image, A, (1, block_size, block_size, 1), padding='SAME')
```
### This can be modified in "train.py" and "model.py" where use the "for" to generate the y. It is time-consuming to use the "for". So if you use the new code, it will be faster. Because it will be processed in GPU instead of CPU.
