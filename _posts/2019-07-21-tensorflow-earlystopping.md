---
title: Tensorflow的EarlyStopping技术
type: post
categories:
- tensorflow
layout: post
date: 2019-07-21
tags: [tensorflow,early stopping]
status: publish
published: true
comments: true
---

# Tensorflow的EarlyStopping参数

![early stopping](https://raw.githubusercontent.com/subaochen/subaochen.github.io/master/images/tensorflow/earlystopping.jpg)

tensorflow的EarlyStopping是callback的一种，允许设置训练提前终止的条件。在下列情况下，往往需要提前终止训练：

* 再训练就过拟合了。
* 再训练下去，也没有明显的改进（就损失函数而言，是没有明显的降低）
* 发生了不收敛的情况（比如学习率设置不当），再训练下去没有意义。

EarlyStopping允许设置的终止训练的条件即参数如下：

* monitor：监控的数据接口。keras定义了如下的数据接口可以直接使用：
  * acc（accuracy），测试集的正确率
  * loss，测试集的损失函数（误差）
  * val_acc（val_accuracy），验证集的正确率
  * val_loss，验证集的损失函数（误差），这是最常用的监控接口，因为监控测试集通常没有太大意义，验证集上的损失函数更有意义。
* patient：对于设置的monitor，可以忍受在多少个epoch内没有改进？patient不宜设置过小，防止因为前期抖动导致过早停止训练。当然也不宜设置的过大，就失去了EarlyStopping的意义了。
* min_delta：评判monitor是否有改进的标准，只有变动范围大于min_delta的monitor才算是改进。对于连续在patient个epoch内没有改进的情况执行EarlyStopping。
* mode：只有三种情况{'min','max','auto'}，分别表示monitor正常情况下是上升还是下降。比如当monitor为acc时mode要设置为'max'，因为正确率越大越好，相反，当monitor为loss时mode要设置为'min'。
* verbose：是否输出更多的调试信息。
* baseline：monitor的基线，即当monitor在基线以上没有改进时EarlyStopping。
* restore_best_weights：当发生EarlyStopping时，模型的参数未必是最优的，即monitor的指标未必处于最优状态。如果restore_best_weights设置为True，则自动查找最优的monitor指标时的模型参数。

# EarlyStopping用法举例

```python
# 监控val_loss，当连续40轮变化小于0.0001时启动early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=40, min_delta=0.0001)
model.fit(callbacks=[es]......)
```

参考：

* [EarlyStopping的官方文档](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping)
* [Tensorflow 2 EarlyStopping教程](https://lambdalabs.com/blog/tensorflow-2-0-tutorial-04-early-stopping/)

