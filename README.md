# Deep Learning for Polarimetric Synthetic Aperture Radar (PolSAR)

Repository for PolSAR deep learning model.

## Convolutional Neural Network (CNN) for PolSAR Image

The [repositroy](https://github.com/eesaber/PolSAR_ML/tree/master/src/sea_ice) implemented CNN models to classified the sea-ice types
in the PolSAR image.
My CNN architecture is modified form the [SegNet](https://arxiv.org/pdf/1511.00561.pdf).
The source code of conventional PolSAR image classification and image analysis were in this [repository](https://github.com/eesaber/Matlab).

### Usage

Different CNN architectures, which varying with number of layer, 2-D size of convolutional filter, were written in
`model_1.py`~`model_6.py`.

* Train a CNN model by
```console
> python3 seg_cnn.py
```
  To change different CNN architecture, import different `model_<>.py`.
