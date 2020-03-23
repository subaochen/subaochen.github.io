---
title: Linux下matplotlib的中文显示问题
type: post
categories:
- 高效工作
layout: post
date: 2020-03-07
tags: [linux,matplotlib,中文]
status: publish
published: true
comments: true
---

matplotlib默认不支持中文，即matplotlib的字体列表中没有把中文字体包括进去，因此中文都会显示为一个一个的方框。下面的步骤在ubuntu 18.04上面是有效的：

1. 找合适的字体库，必须是ttf格式的，ttc的不行。当然，ttc可以转换为ttf格式，比如[这里](https://www.files-conversion.com/font/ttc)可以在线转换。我的办法比较笨，首先安装wqy的微米黑字体（ttc的），然后转换为ttf。

1. 找到matplotlib的字体所在目录，方法是执行：

   ```
   In [1]: import matplotlib
   In [2]: from matplotlib.font_manager import findfont, FontProperties
   In [3]: findfont(FontProperties(family=FontProperties().get_family()))
   Out[3]: '/home/subaochen/anaconda3/envs/py36/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf'
   ```

   可以看出，在我这里，matplotlib把ttf字体放到了`/home/subaochen/anaconda3/envs/py36/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf/`这个目录下，因此把第一步生成的wqy微米黑ttf文件也复制到这里。

1. 修改matplotlib的配置文件。通过第二步，我们已经知道matplotlib的ttf文件所在目录，matplotlib的配置文件matplotlibrc就在上一级目录，打开配置文件，修改关于sans-serif字体的设置和minus的设置部分如下：

```
font.family         : sans-serif
font.sans-serif     : WenQuanYi Micro Hei, DejaVu Sans, Bitstream Vera Sans, Computer Modern Sans Serif, Lucida Grande, Verdana, Geneva, Lucid, Arial, Helvetica, Avant Garde, sans-serif
axes.unicode_minus  : False # 防止-号变成方框
```
经过以上的三个步骤，matplotlib就可以直接使用中文了。当然，需要重新启动python解释器，或者在jupyter中restart kernel。

   

   

   

