---
title: XTP在linux下面的编译
type: post
categories:
- database
layout: post
date: 2020-03-20
tags: [XTP,linux,boost]
status: publish
published: true
comments: true
---

中泰证券的[XTP](https://xtp.zts.com.cn/)似乎在国内备受推崇，于是拉下来其[python开发包](https://github.com/ztsec/xtp_api_python)看了一下，结果不断踩坑备受折磨，在bobo liu同学的帮助下才成功编译。简单的说，就是如何在Linux环境下重新编译boost和XTP的python封装？

**第一个坑**：编译boost。XTP自带预编译版本是需要boost 1.66版本支持的，但是anaconda3的源却恰恰丢失了这个版本的boost，于是干脆下载了最新版的boost 1.72.0重新编译。XTP的文档中说明了如何在Linux环境下编译boost，但是那个文档有两个地方不妥：第一，不需要重新编译python；第二，重新编译boost时给出的选项不对，应该是：`./b2 --toolset=gcc-7  --with-python include="/home/subaochen/anaconda3/envs/xtp/include/python3.6m/" --with-thread --with-date_time  --with-chrono --with-system`，这里需要注意几点：

* 查找自己系统的gcc版本，我这里是ubuntu 18.04，gcc版本为gcc-7
* 我使用了anaconda的虚拟环境，因此include应该设置为虚拟环境xtp的include路径，如上所示。
* 原文档漏掉了--with-system选项，后面编译xtp的时候会用到。

**第二个坑**：文档中说的很模糊，编译好的boost库文件放到哪里？简单的办法是在.bashrc中增加如下的两行（注意，我把boost 1.72.0放到了devel目录下，并符号链接到了boost）：

```
export BOOST_INCLUDE=$HOME/devel/boost
export BOOST_LIB=$HOME/boost/stage/lib
```

**第三个坑**：编译XTP，这个坑更大，涉及到三个文件的修改和一个环境的配置。

第一，修改XTP的source目录下Linux版本的CMakeLists.txt文件，找到对应行改为：

```
....
    set(PYTHON_LIBRARY /home/subaochen/anaconda3/envs/xtp/lib/)
    set(PYTHON_INCLUDE_PATH 
    /home/subaochen/anaconda3/envs/xtp/include/python3.6m/)
....
    set(BOOST_ROOT   /home/subaochen/devel/boost)
    find_package(Boost 1.72.0 COMPONENTS python36 thread date_time system chrono REQUIRED)
```

也就是说，设置合适的环境变量，修改boost的版本号，以及python模块的名字为python36（具体查看编译boost时生成的库文件，位于boost/stage/lib）。

<!--第二，还需要修改vnxtpquote.cpp的第1954行为：-->

```
class_<QuoteApiWrap, boost::shared_ptr<QuoteApiWrap>, boost::noncopyable>("QuoteApi")
```

<!--这个看不太懂，根据bobo liu同学的解释，应该是boost会直接读取python的类名作为数据类型，因此需要在泛型中强制类型转换为QuoteApi类型。同样的道理，也需要修改vnxtptrader.cpp文件的对应位置。-->

第二，修改vnxtpquote.h/vnxtptrader.h中关于mutex的定义，原始代码的命名空间没搞清楚，应该为：`boost::mutex`。

第三，编译XTP完成后生成的两个so文件以及XTP的c++原生SDK的两个lib库（也是so文件，libxtp...so，位于bin/Linux目录下），我采取了简单的办法，放到了一个单独的目录下，并设置了两个环境变量，以便任何环境下都能找到这几个so：

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/subaochen/devel/xtplib
export PYTHONPATH=$PYTHONPATH:$HOME/devel/xtplib
```



排除了上面几个坑，应该就可以愉快的执行quotetest.py和tradertest.py了，量化交易模拟愉快:-)