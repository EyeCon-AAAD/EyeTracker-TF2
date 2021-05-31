# EyeTracker-TF2

Tensorflow 2 and Keras Implementation of _EyeTracking for Everyone_ paper by _Torralba et al_.

## Configuration
Model was trained on the small dataset of 48K training samples and 5000 validation samples found [here](http://hugochan.net/download/eye_tracker_train_and_val.npz)
with the following configuration:
* Python 3.8
* Tensorflow 2.5.0
* Keras 2.4.3
* Numpy 1.19.5

## Results
Results are obtained on 64x64 images. The results are expressed in terms of Mean Absolute Error (MAE) and Standard 
Deviation. The network was tested on the aforementioned dataset.

| Input Size    | MAE           | STD           | MSE Loss
| ------------- | -----         | -----         | --------
| 64x64         | 0.89, 0.99    | 1.09, 1.22    | 2.2281

## Acknowledgments
This work wouldn't be accomplished without the contribution of [hugochan](https://github.com/hugochan/Eye-Tracker) 
and [gdubrg](https://github.com/gdubrg/Eye-Tracking-for-Everyone)

