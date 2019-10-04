# Diversify and Match 

### Acknowledgment

The implementation is built on the pytorch implementation of Faster RCNN [jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch)


## Preparation
1. Clone the code and create a folder
```
git clone https://github.com/TKKim93/DiversifyAndMatch.git
cd faster-rcnn.pytorch && mkdir data
```

2. Build the Cython modules
```Shell
cd DiversifyAndMatch/lib
make
``` 

### Prerequisites

* Python 2.7 or 3.6
* Pytorch 0.4.0 
* CUDA 8.0 or higher
* cython, cffi, opencv-python, scipy, easydict, matplotlib, pyyaml

### Pretrained Model
You can download pretrained VGG and ResNet101 from [jwyang's repository](https://github.com/jwyang/faster-rcnn.pytorch). The default location in my code is './data/pretrained_model/'.

### Repository Structure
```
DivMatch
├── cfgs
├── data
│   ├── pretrained_model
├── datasets
│   ├── clipart
│   │   ├── Annotations
│   │   ├── ImageSets
│   │   ├── JPEGImages
│   ├── clipart_CP
│   ├── clipart_CPR
│   ├── clipart_R
│   ├── comic
│   ├── comic_CP
│   ├── comic_CPR
│   ├── comic_R
│   ├── Pascal
│   ├── watercolor_CP
│   ├── watercolor_CPR
│   ├── watercolor_R
├── lib
├── models (save location)
```

## Example
### All at once


### Diversification stage
### Matching stage
Here is an example of adapting from Pascal VOC to Clipart1k:
1. You can prepare the Pascal VOC datasets from [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn) and the Clipart1k dataset from [cross-domain-detection](https://github.com/naoto0804/cross-domain-detection) in VOC data format.
2. Shift the source domain through domain shifter. Basically, I used a residual generator and a patchGAN discriminator. For the short cut, you can download some examples of shifted domains (Link) and put these datasets into data folder.
3. Train the object detector through MRL for the Pascal -> Clipart1k adaptation task.
```
    python train.py --dataset clipart --net vgg16 --cuda
```
4. Test the model
```
    python test.py --dataset clipart --net vgg16
```

## Downloads
* Shifted domains for Clipart1k: [Clip_CP](https://drive.google.com/open?id=1k1Yn1IMwffCFE_MTfC4WvlajWS9a783G), [Clip_R](https://drive.google.com/open?id=1whHjLyqL3-mkYoXXhAFDu7rYzoe9MoM_), [Clip_CPR](https://drive.google.com/open?id=1Tq3pQRwCOezyRtxf69ZVO8fUA_E64Tbt)
* Shifted domains for Watercolor2k: [Wat_CP](https://drive.google.com/open?id=1i_q6ySLtE3353Wep5Gz32YhEi0ahuEtD), [Wat_R](https://drive.google.com/open?id=1NTq0GN9H8nnl2D8A5pbye890HjWKsP8Q), [Wat_CPR](https://drive.google.com/open?id=1MTIvekWwnUih1o1oYZ-qbkRsAti3utos)
* Shifted domains for Comic2k: [Com_CP](https://drive.google.com/open?id=1JJPRmSUaIW_FC57sguNHFwnuOk9do3Vc), [Com_R](https://drive.google.com/open?id=1ixrslHKiluiKWppwzFszXYlsEXWpWVru), [Com_CPR](https://drive.google.com/open?id=1oGcSwNpTL-IJ0G3Ao71Ke4ZPfxMYj8In)
* shifted domains for Cityscapes: City_CP, City_R, City_CPR
