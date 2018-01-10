# Object-Detection-Development

### Outline
1. Classical papers
2. Benchmarks
3. Great open code
4. Extra Reading notes
5. Very Recent Works


###  1. Classical  papers
- 2013    **RCNN** --[Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/pdf/1311.2524)
- 2013   ----Deep Neural Networks for Object Detection
- 2014   **SPPnet** --[Spatial Pyramid Pooling in Deep Convolutional Networks for Visual](https://arxiv.org/pdf/1406.4729) Recognition-sppnet
- 2015 [Cascaded Sparse Spatial Bins for Efficient and Effective Generic Object Detection](https://arxiv.org/pdf/1504.07029)
- 2015 **[Fast R-CNN](https://arxiv.org/pdf/1504.08083)**
- 2016 **Faster R-CNN** --[Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/pdf/1506.01497)
- 2016 Inside-Outside Net_Detecting Objects in Context with Skip Pooling and Recurrent Neural Networks
- 2016 **R-FCN** --[Object Detection via Region-based Fully Convolutional Networks](https://arxiv.org/pdf/1605.06409)
- 2016 **SSD** --[SSD: Single-Shot MultiBox Detector](https://arxiv.org/pdf/1512.02325)

- 2016 **YOLO9000** --[YOLO9000: better,faster,stronger](https://arxiv.org/pdf/1612.08242)
- 2016 **YOLO** --[You Only Look Once-Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640)
- 2017 **A-Fast-RCNN** --[A-Fast-RCNN_Hard Positive Generation via Adversary for Object Detection](hhttps://arxiv.org/pdf/1704.03414)
- 2017 [Speed_Accuracy trade-offs for modern convolutional object detectors](https://arxiv.org/pdf/1611.10012)
- 2017 **DeformableCNN** --[Deformable Convolutional Networks](https://arxiv.org/pdf/1703.06211)
- 2017 **DSOD** --[DSOD: Learning Deeply Supervised Object Detectors from Scratch](https://arxiv.org/pdf/1708.01241)
- 2017 **Focal Loss** --[Focal Loss for Dense Object Detection](https://arxiv.org/pdf/1708.02002)
- 2017 **Mask R-CNN** --[Mask R-CNN](https://arxiv.org/pdf/1703.06870)

- 2017 [Light-Head R-CNN_In Defense of Two-Stage Object Detector](https://arxiv.org/pdf/1711.07264)

- TBA

--------


### 2.Dataset&Benchmarks
- Pascal VOC
- ImageNet
- MSCOCO
- KITTI
- CityScapes
- TT100K
- TBA


--------

### 3.Great open code

- Faster-rcnn [MatCAFFE](https://github.com/ShaoqingRen/faster_rcnn),[PyCAFFE](https://github.com/rbgirshick/py-faster-rcnn),[Tensorflow](https://github.com/smallcorgi/Faster-RCNN_TF),[PyTorch](https://github.com/longcw/faster_rcnn_pytorch),[mx-Net](https://github.com/precedenceguo/mx-rcnn)
- Deformable-NN Det [mx-Net](https://github.com/msracver/Deformable-ConvNets),[PyTorch](https://github.com/oeway/pytorch-deform-conv)
- R-FCN [MatCAFFE](https://github.com/daijifeng001/R-FCN),[PyCAFFE](https://github.com/daijifeng001/R-FCN),[PyTorch](https://github.com/PureDiors/pytorch_RFCN),[CAFFE-MultiGPU](https://github.com/bharatsingh430/py-R-FCN-multiGPU)
- SSD [PyTorch](https://github.com/amdegroot/ssd.pytorch),[mx-net](https://github.com/amdegroot/ssd.pytorch)
- YOLO [tensorflow](https://github.com/gliese581gg/YOLO_tensorflow),[PyTorch](),[CAFFE](https://github.com/philipperemy/yolo-9000),[CAFFE](https://github.com/xingwangsfu/caffe-yolo)
- YOLO2 [PyTorch](https://github.com/longcw/yolo2-pytorch),[PyTorch](https://github.com/marvis/pytorch-yolo2)
- DSOD [PyCAFFE](https://github.com/szq0214/DSOD)
- 
- Vehicle Detection & Tracking [1](https://github.com/kkufieta/CarND-Vehicle-Detection), [2](https://github.com/LeotisBuchanan/udacity_vehicle_detection), [3]

- Vehicle Detection and Tracking using HOG, CNN [1](https://github.com/xmprise/Vehicle_Detection_and_Tracking)
- SSD for KITTI [1](https://github.com/manutdzou/KITTI_SSD)
- SqueezeDet on KITTI [1](https://github.com/fregu856/2D_detection)
- Faster-rcnn for KITTI [1](https://github.com/manutdzou/KITTI_FRC_detection)
- 3D CNN for KITTI [1](https://github.com/yukitsuji/3D_CNN_tensorflow)
- lane detection based on KITTI model [1](https://github.com/catpanda/lane_detection)
- eval KITTI results [1](https://github.com/cguindel/eval_kitti)
- DIDI detection competition [1](https://github.com/omgteam/Didi-competition-solution) [2](https://github.com/sir-siemens/team-007)
- Udacity Self-driving Car [1](https://github.com/CarND-Capstone-Defender/car-nd-capstone) [2](https://github.com/byronrwth/Udacity-SelfDrivingCar-Term2) [YOLO](https://github.com/aashay96/YOLO-Udacity) [traffic light detection](https://github.com/awoodacrew/tldetect)
[final project](https://github.com/AndysDeepAbstractions/Early_Birds_CarND-Capstone)

- [RFBNet](https://github.com/ruinmessi/RFBNet)
- TBA
- 

------

### 4.Extra Reading Notes

- Detection Summary Until 17.1 [csdn](http://blog.csdn.net/zhang11wu4/article/details/53967688)
- Relation Networks for Object Detection [zhihu](https://zhuanlan.zhihu.com/p/31742364)
- SSD details [zhihu](https://zhuanlan.zhihu.com/p/31427288)

- Multi-view 3D detection [csdn](http://blog.csdn.net/williamyi96/article/details/78043014)
[csdn](https://www.baidu.com/link?url=7MyT1jpd6AUtAcQ6wPKZAkAGSCySPSstaKNPJW2d__E2DVMqMS7Gkg3AtwhkLlDlATmB4c1-zx1B9sAllKpfxiTRVteQYnONnA1DTnahB8y&wd=&eqid=a90695100000bc08000000035a561a78)

------

### 5.Very Recent Works

#### a. For object detection
- [Feature Selective Networks for Object Detection](https://arxiv.org/pdf/1711.08879.pdf)
- [Frustum PointNets for 3D Object Detection from RGB-D Data](https://arxiv.org/pdf/1711.08488.pdf)
- [MegDet: A Large Mini-Batch Object Detector](https://arxiv.org/pdf/1711.07240.pdf)
- [Deep Image Prior](https://arxiv.org/pdf/1711.10925.pdf)
- [Non-Local](https://arxiv.org/pdf/1711.07971.pdf)
- [An Analysis of Scale Invariance in Object Detection – SNIP](https://arxiv.org/pdf/1711.08189.pdf)
- [Receptive Field Block Net for Accurate and Fast Object Detection](https://arxiv.org/pdf/1711.07767.pdf)
- [Single-Shot Refinement Neural Network for Object Detection](https://arxiv.org/pdf/1711.06897.pdf)
- [VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection](https://arxiv.org/pdf/1711.06396.pdf)
- [Contextual Object Detection with a Few Relevant Neighbors](https://arxiv.org/pdf/1711.05705.pdf)
- [Dynamic Zoom-in Network for Fast Object Detection in Large Images](https://arxiv.org/pdf/1711.05187.pdf)
- [A Taught-Obesrve-Ask (TOA) Method for Object
Detection with Critical Supervision](https://arxiv.org/pdf/1711.01043.pdf)
- [Single Multi-feature detector for Amodal 3D Object Detection in RGB-D Images](https://arxiv.org/pdf/1711.00238.pdf)
- [Cascade Region Proposal and Global Context for Deep
Object Detection](https://arxiv.org/pdf/1710.10749.pdf)

- [Relation networks](https://arxiv.org/pdf/1711.11575.pdf)
- 

#### b. For car detection especially 
- TBA
