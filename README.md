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
    
### Prerequisites

* Python 2.7 or 3.6
* Pytorch 0.4.0 
* CUDA 8.0 or higher

## Example
Here is an example of adapting from Pascal VOC to Clipart1k:
1. You can prepare the Pascal VOC datasets from [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn) and the Clipart1k from [naoto0804/cross-domain-detection](https://github.com/naoto0804/cross-domain-detection) in VOC data format.
2. Shift the source domain through domain shifter. Basically I used a residual generator and a patchGAN discriminator. For the short cut, you can download some examples of shifted domains.
3.
