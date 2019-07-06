[TOC]




## 图像算法大全：


```

滤波、边缘、角点、图像分割、轮廓检测、霍夫检测、形态学操作【腐蚀膨胀开闭】、直方图均衡化

八链码： 描述曲线或边界的方法，表示曲线和区域边界

凸包：不规则图像凸包、最小规则图形凸包、椭圆拟合、逼近多边形曲线

傅里叶变换·频域滤波：高斯高/低通滤波

图像压缩 与 编码、小波变换


特征提取：

Surf、Sift、LBP、Hog、ORB

角点系：Harris、Tomasi、亚像素角点

边缘系：DoG一阶边缘提取/Log拉普拉斯-高斯边缘提取

MSER/Fast/Gist/Freak/Akaze/Brisk/Star/GFTT/Dense/SimpleBlob


特征匹配：

FLANN / BFMatcher匹配



-----------------------

特定算法：

Otsu最优阈值检测、非极大值抑制、Fisher最优分割法-聚类、匈牙利算法-二分图最大匹配、MB/MB2 骨架提取算法

KD-树最近邻搜索、Kohoen特征图-自组织神经网络、Hopfield自组织神经网络

Ransac随机采样一致性算法、PDM形状、近似配准、ASM配准、AAM建模&配准


轮廓形状描述子：

	链码、边界长度、曲率(HK2003算法)、弯曲能量、签名、弦分布

	傅里叶形状描述子、B样条、形状不变量

几何区域描述子：

	面积、欧拉数、投影、宽高、离心率、细长度、矩形度、方向、紧致度

	统计矩、凸包、细化骨架、区域分解、区域邻近图

```


#### 图像基础：


​	灰度图：图像混合：

​	阈值化：

​	直方图：灰度直方图、H-S直方图

​	重映射(几何变换-镜像、反转...)：反向投影(通过模板直方特征在原图中对定位模板)：

​	图像金字塔：高斯-、拉普拉斯-、上采样、下采样

​	滤波算子：

​		高斯滤波、中值滤波、均值滤波、方框(盒式)滤波、双边滤波、低/高通滤波、引导(导向)滤波

​		边缘检测算子：Canny算子、Sobel算子、Laplacian算子、Scharr滤波

​		频域滤波：Gabor滤波、Frangi滤波

​		时域滤波：

​			限幅滤波、中值滤波、均值滤波、滑动平均滤波、中值平均滤波、限幅平均滤波

​			一阶滞后滤波、加权递推平均滤波、消抖滤波

​		Lanczos滤波、卡尔曼滤波、粒子滤波、逆滤波、维纳滤波


​	绘制：线、矩、圆、FloodFill 漫水填充 

​	轮廓检测：

​		形状检测：霍夫直线 & 圆检测、曲线检测

​		凸包、椭圆拟合、多边形逼近

​	模板匹配：平方差法、相关法、系数法...

​	特征匹配：暴力匹配、Flann匹配

​	矩特征：矩的计算、面积周长计算、质心、一二三阶矩

​	距离变换：透视 & 仿射变换：

​	八链码、傅里叶变换、小波变换


#### 形态学操作：


​	腐蚀，膨胀

​	开，闭

​	形态学梯度、顶帽、黑帽


#### 角点检测：


​	Harris、Tomasi、亚像素角点


#### 边缘检测：


​	Canny 边缘检测：DoG一阶边缘提取：Log拉普拉斯-高斯边缘提取：


#### 图像分割：


​	阈值法、基于区域、基于边缘、基于基因编码、基于小波变换

​	分水岭：

​	邻域边缘跟踪： 

​	邻域区域生长：

​	最小割最大流：

​	GraphCut & GrabCut：

​	聚类法：MeanShift


#### 特征提取：


​	(关键点 kp、描述子 [x]、特征图)

​	Hog：Sift：Surf：ORB：MSER：LBP：Retina：

​	Fast：Brisk：Star：GFTT：Dense：SimpleBlob：Akaze：Gist：Freak：

​	纹理：Gabor：GLCM：GMRF：GLDS：FD：


#### 对象跟踪：


​	光流法、背景消除建模、聚类法


------


#### 人脸对齐：


​	ASM-Active Shape Model、AAM-Active Appearance Model

​	CLM-Constrained Local Model、SDM-Supervised Descent Method


ESR	Explicit Shape Regression		cvpr2012

ERT	Ensemble of Regression Trees		cvpr2014

LBF	Regressing Local Binary Features

CFSS	Coarse-to-Fine Shape Searching		cvpr2015

TCDCN	Tasks-Constrained Deep Convolutional Network

MTCNN	Multi-task convolutional neural networks

DAN	Deep Alignment Network 			cvpr2017

LAB	清华-商汤 人脸对齐2018			cvpr2018[TOC]





## 图像算法大全：



```


滤波、边缘、角点、图像分割、轮廓检测、霍夫检测、形态学操作【腐蚀膨胀开闭】、直方图均衡化


八链码： 描述曲线或边界的方法，表示曲线和区域边界


凸包：不规则图像凸包、最小规则图形凸包、椭圆拟合、逼近多边形曲线


傅里叶变换·频域滤波：高斯高/低通滤波


图像压缩 与 编码、小波变换



特征提取：


Surf、Sift、LBP、Hog、ORB


角点系：Harris、Tomasi、亚像素角点


边缘系：DoG一阶边缘提取/Log拉普拉斯-高斯边缘提取


MSER/Fast/Gist/Freak/Akaze/Brisk/Star/GFTT/Dense/SimpleBlob



特征匹配：


FLANN / BFMatcher匹配




-----------------------


特定算法：


Otsu最优阈值检测、非极大值抑制、Fisher最优分割法-聚类、匈牙利算法-二分图最大匹配、MB/MB2 骨架提取算法


KD-树最近邻搜索、Kohoen特征图-自组织神经网络、Hopfield自组织神经网络


Ransac随机采样一致性算法、PDM形状、近似配准、ASM配准、AAM建模&配准



轮廓形状描述子：


	链码、边界长度、曲率(HK2003算法)、弯曲能量、签名、弦分布


	傅里叶形状描述子、B样条、形状不变量


几何区域描述子：


	面积、欧拉数、投影、宽高、离心率、细长度、矩形度、方向、紧致度


	统计矩、凸包、细化骨架、区域分解、区域邻近图


```



#### 图像基础：



​	灰度图：图像混合：


​	阈值化：


​	直方图：灰度直方图、H-S直方图


​	重映射(几何变换-镜像、反转...)：反向投影(通过模板直方特征在原图中对定位模板)：


​	图像金字塔：高斯-、拉普拉斯-、上采样、下采样


​	滤波算子：


​		高斯滤波、中值滤波、均值滤波、方框(盒式)滤波、双边滤波、低/高通滤波、引导(导向)滤波


​		边缘检测算子：Canny算子、Sobel算子、Laplacian算子、Scharr滤波


​		频域滤波：Gabor滤波、Frangi滤波


​		时域滤波：


​			限幅滤波、中值滤波、均值滤波、滑动平均滤波、中值平均滤波、限幅平均滤波


​			一阶滞后滤波、加权递推平均滤波、消抖滤波


​		Lanczos滤波、卡尔曼滤波、粒子滤波、逆滤波、维纳滤波



​	绘制：线、矩、圆、FloodFill 漫水填充 


​	轮廓检测：


​		形状检测：霍夫直线 & 圆检测、曲线检测


​		凸包、椭圆拟合、多边形逼近


​	模板匹配：平方差法、相关法、系数法...


​	特征匹配：暴力匹配、Flann匹配


​	矩特征：矩的计算、面积周长计算、质心、一二三阶矩


​	距离变换：透视 & 仿射变换：


​	八链码、傅里叶变换、小波变换



#### 形态学操作：



​	腐蚀，膨胀


​	开，闭


​	形态学梯度、顶帽、黑帽



#### 角点检测：



​	Harris、Tomasi、亚像素角点



#### 边缘检测：



​	Canny 边缘检测：DoG一阶边缘提取：Log拉普拉斯-高斯边缘提取：



#### 图像分割：



​	阈值法、基于区域、基于边缘、基于基因编码、基于小波变换


​	分水岭：


​	邻域边缘跟踪： 


​	邻域区域生长：


​	最小割最大流：


​	GraphCut & GrabCut：


​	聚类法：MeanShift



#### 特征提取：



​	(关键点 kp、描述子 [x]、特征图)


​	Hog：Sift：Surf：ORB：MSER：LBP：Retina：


​	Fast：Brisk：Star：GFTT：Dense：SimpleBlob：Akaze：Gist：Freak：


​	纹理：Gabor：GLCM：GMRF：GLDS：FD：



#### 对象跟踪：



​	光流法、背景消除建模、聚类法



------



#### 人脸对齐：



​	ASM-Active Shape Model、AAM-Active Appearance Model


​	CLM-Constrained Local Model、SDM-Supervised Descent Method



ESR	Explicit Shape Regression		cvpr2012


ERT	Ensemble of Regression Trees		cvpr2014


LBF	Regressing Local Binary Features


CFSS	Coarse-to-Fine Shape Searching		cvpr2015


TCDCN	Tasks-Constrained Deep Convolutional Network


MTCNN	Multi-task convolutional neural networks


DAN	Deep Alignment Network 			cvpr2017


LAB	清华-商汤 人脸对齐2018			cvpr2018

