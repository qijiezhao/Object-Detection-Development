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
- 2016 **PVANet** --[PVANET: Deep but Lightweight Neural Networks for
Real-time Object Detection](https://arxiv.org/pdf/1608.08021v1.pdf)
- 2016 **YOLO9000** --[YOLO9000: better,faster,stronger](https://arxiv.org/pdf/1612.08242)
- 2016 **YOLO** --[You Only Look Once-Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640)
- 2017 **A-Fast-RCNN** --[A-Fast-RCNN_Hard Positive Generation via Adversary for Object Detection](hhttps://arxiv.org/pdf/1704.03414)
- 2017 [Speed_Accuracy trade-offs for modern convolutional object detectors](https://arxiv.org/pdf/1611.10012)
- 2017 **DSSD** --[DSSD : Deconvolutional Single Shot Detector](https://arxiv.org/pdf/1701.06659.pdf)
- 2017 **DeformableCNN** --[Deformable Convolutional Networks](https://arxiv.org/pdf/1703.06211)
- 2017 **FPN** --[Feature Pyramid Networks for Object Detection](https://arxiv.org/pdf/1612.03144.pdf)
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
- SSD [PyTorch](https://github.com/amdegroot/ssd.pytorch),[mx-net](https://github.com/amdegroot/ssd.pytorch),[SSD512-pytorch](https://github.com/lopuhin/ssd.pytorch)
- YOLO [tensorflow](https://github.com/gliese581gg/YOLO_tensorflow),[PyTorch](),[CAFFE](https://github.com/philipperemy/yolo-9000),[CAFFE](https://github.com/xingwangsfu/caffe-yolo)
- YOLO2 [PyTorch](https://github.com/longcw/yolo2-pytorch),[PyTorch](https://github.com/marvis/pytorch-yolo2)
- DSOD [PyCAFFE](https://github.com/szq0214/DSOD)
- RetinaNet [pytorch](https://github.com/c0nn3r/RetinaNet)

-
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
- MV3D [tensorflow](https://github.com/bostondiditeam/MV3D)
- Depth and Ego-motion prediction [pytorch](https://github.com/ClementPinard/SfmLearner-Pytorch)
- Deeply fuse Net [pytorch](https://github.com/zlmzju/fusenet)

------

### 4.Extra Reading Notes

- Detection Summary Until 17.1 [csdn](http://blog.csdn.net/zhang11wu4/article/details/53967688)
- Relation Networks for Object Detection [zhihu](https://zhuanlan.zhihu.com/p/31742364)
- SSD details [zhihu](https://zhuanlan.zhihu.com/p/31427288)

- Multi-view 3D detection [csdn](http://blog.csdn.net/williamyi96/article/details/78043014)
[csdn](https://www.baidu.com/link?url=7MyT1jpd6AUtAcQ6wPKZAkAGSCySPSstaKNPJW2d__E2DVMqMS7Gkg3AtwhkLlDlATmB4c1-zx1B9sAllKpfxiTRVteQYnONnA1DTnahB8y&wd=&eqid=a90695100000bc08000000035a561a78)

- semi-supervised learning on Image [zhihu](https://zhuanlan.zhihu.com/p/32658795?group_id=932637377097244672)
- CycleGAN,DualGAN,DiscoGAN [zhihu](https://zhuanlan.zhihu.com/p/32800494?utm_source=wechat_session&utm_medium=social)
- 人脸遮挡问题的处理 [旷视,wechat](http://mp.weixin.qq.com/s/QJm7YoCYmiF0dX8uac5w4Q)
- 相机矩阵，内参数与外参数 [csdn](http://blog.csdn.net/zb1165048017/article/details/71104241)
------

### 5.Very Recent Works

#### a. For 2D object detection
- [Feature Selective Networks for Object Detection](https://arxiv.org/pdf/1711.08879.pdf)

- [MegDet: A Large Mini-Batch Object Detector](https://arxiv.org/pdf/1711.07240.pdf)
- [Deep Image Prior](https://arxiv.org/pdf/1711.10925.pdf)

- [An Analysis of Scale Invariance in Object Detection – SNIP](https://arxiv.org/pdf/1711.08189.pdf)
- [Receptive Field Block Net for Accurate and Fast Object Detection](https://arxiv.org/pdf/1711.07767.pdf)
- [Single-Shot Refinement Neural Network for Object Detection](https://arxiv.org/pdf/1711.06897.pdf)

- [Contextual Object Detection with a Few Relevant Neighbors](https://arxiv.org/pdf/1711.05705.pdf)
- [Dynamic Zoom-in Network for Fast Object Detection in Large Images](https://arxiv.org/pdf/1711.05187.pdf)
- [A Taught-Obesrve-Ask (TOA) Method for Object
Detection with Critical Supervision](https://arxiv.org/pdf/1711.01043.pdf)

- [Cascade Region Proposal and Global Context for Deep Object Detection](https://arxiv.org/pdf/1710.10749.pdf)

- [Relation networks](https://arxiv.org/pdf/1711.11575.pdf)
- [Single Shot Text Detector with Regional Attention](https://arxiv.org/pdf/1709.00138.pdf)
- [DSSD](https://arxiv.org/pdf/1701.06659.pdf)
- [RRC](https://arxiv.org/pdf/1704.05776.pdf)

#### b. For 3D object detection

- [House 3D, a dataset](https://github.com/facebookresearch/House3D)
- [Frustum PointNets for 3D Object Detection from RGB-D Data](https://arxiv.org/pdf/1711.08488.pdf)
- [VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection](https://arxiv.org/pdf/1711.06396.pdf)
- [Single Multi-feature detector for Amodal 3D Object Detection in RGB-D Images](https://arxiv.org/pdf/1711.00238.pdf)
- [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/pdf/1612.00593.pdf)
- [3D Object Proposals using Stereo Imagery for Accurate Object Class Detection](https://arxiv.org/pdf/1608.07711.pdf)
- [SSD-6D: Making RGB-Based 3D Detection and 6D Pose Estimation Great Again](https://arxiv.org/pdf/1711.10006.pdf)

#### c. For car detection especially 
- [PointFusion: Deep Sensor Fusion for 3D Bounding Box Estimation](https://arxiv.org/pdf/1711.10871.pdf)
- [Joint 3D Proposal Generation and Object Detection from View Aggregation](https://arxiv.org/pdf/1712.02294.pdf)
- [Fusing Bird View LIDAR Point Cloud and Front
View Camera Image for Deep Object Detection](https://arxiv.org/pdf/1711.06703.pdf)
- [A Joint 3D-2D based Method for Free Space Detection on Roads](https://arxiv.org/pdf/1711.02144.pdf)
- [PointCNN](https://arxiv.org/pdf/1801.07791.pdf)
-
-

#### d. Other useful tools

- [Non-Local](https://arxiv.org/pdf/1711.07971.pdf)
- [Left-Right Skip-DenseNets for Coarse-to-Fine Object Categorization](https://arxiv.org/pdf/1710.10386.pdf)
- [Learning by Asking Questions](https://arxiv.org/pdf/1712.01238.pdf)
