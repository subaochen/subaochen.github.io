---
title: Linux下面双面打印的方法（FS-1020MFP）
type: post
categories:
- other
layout: post
date: 2019-07-14
tags: [linux,print]
status: publish
published: true
use_math: false
comments: true
---

每次双面打印都要实验半天浪费好多纸张才搞明白怎么弄:-(，Linux下面的FS-1020MFP打印驱动还是有点弱啊。好记性不如烂笔头，何况老年痴呆初期患者。

步骤有三：

1. 打印奇数页，打印机选项如下图所示：
   ![](https://raw.githubusercontent.com/subaochen/subaochen.github.io/master/images/office/double-print-odd.png)

2. 将打印好的奇数页**不要反面**，直接调转180度再次塞进送纸器。

3. 打印偶数页，打印机选项如下图所示，注意除了选择even pages之外，还要选择**“反向”**！

![](https://raw.githubusercontent.com/subaochen/subaochen.github.io/master/images/office/double-print-even.png)