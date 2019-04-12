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
