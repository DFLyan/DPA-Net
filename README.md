# DPA-Net
## Dataset
You can use crop.m to generate training data.

## Use trained model
You can download the trained model from [Google Drive](https://drive.google.com/open?id=1-fvKrbUg7Q0wWhiwpUpXbsBf4SQl50c3). The put them in the file folder--checkpoint.

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

You can change the setting in config.py, such as MR(measurement ratio) and so on. The files of trained model parameter and train data will be uploaded soon.
