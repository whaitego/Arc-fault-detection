# 机器学习电弧检测项目
## 文件介绍
**arc-fault detection.ipynb**:主程序，包含整个代码流程
**changepoint.py**：用于变点检测方法
**ML.py**:用于机器学习方法
**Visualization.py**:用于可视化结果图，包含热图、雷达图等方法
**functions.py**:包含一些小函数
**merge_to_gif.gif**:运行该项目得到的gif图像，用于展示
**requirements.txt**:包含了该项目依赖的对应版本的python包
**README.md**:说明文档

**注**：请确保上述的.ipynb和.py在同一文件目录下，不然运行会报错
**注+**：要正确运行arc-fault detection.ipynb还需要GetData.py文件，该文件位于**IAED数据集**中,请参考下面对数据集的介绍，需将**GateData.py**单独放入和.ipynb、.py同一文件目录下方可正常运行

## 项目背景

本项目是作为编程基础作业的一部分而开发的。在这门课程中，我们被要求使用机器学习方法来解决实际问题。本项目的目标是利用不同的机器学习算法进行电弧检测，并对这些方法的性能进行评估和比较。

## 目的和范围

该项目旨在展示如何实施和评估不同的机器学习技术。作为一个学术作业，它的设计和实现旨在符合编程基础的课程要求，并展示对学习材料的理解和应用。
请注意，本项目是在学术环境下开发的，可能仅作为概念验证或原型展示。

## 简介
本项目使用机器学习和变点检测方法进行电弧检测，旨在通过分析和处理数据，准确地识别电弧事件。项目中包含.ipynb和.py文件，其中.ipynb文件包含了项目的核心代码和流程，而.py文件则定义了一些在.ipynb中用到的类和函数。

## 特点
**多种机器学习方法**：项目支持多种机器学习算法，包括支持向量机（SVM）、决策树（Decision Tree）等。
**变点检测方法**：包含了基于Hampel滤波器原理的MAD检测方法
**性能评估**：能够计算和展示多个性能指标，如假正率(FPN)、假负率(FNN)、精确率(PRECISION)、召回率(RECALL)等。
**可视化比较**：提供了不同机器学习方法和不同性能指标之间比较的可视化展示。

## 安装
git clone或者直接下载压缩包均可

## 使用方法
请确保将.py文件和.ipynb置于同一文件路径下，使用jupyter notebook运行**arc-fault detection.ipyn**即可

## 运行环境

- 操作系统：Windows, MacOS, Linux
- Python版本：Python 3.8及以上

## 安装依赖（非必须）

本项目依赖于多个Python库，要安装这些依赖，请运行以下命令：
pip install -r requirements.txt

`requirements.txt`文件包含了所有必要的库及其版本。


## 数据集

本项目使用了IAED，这是一个公开的数据集，由TianHuiyun创建和维护。该数据集对于本项目的研究至关重要，我们在此对原作者表示感谢。

数据集链接：[github](https://github.com/inteverdata/IAED)

请注意，本项目不包含数据集文件，你需要从上述链接下载。

下载数据集并解压到**arc-fault detection.ipyn**文件所在目录下即可使用

## 致谢

特别感谢TianHuiyun提供的IAED数据集，它对我们的研究工作提供了宝贵的支持。

