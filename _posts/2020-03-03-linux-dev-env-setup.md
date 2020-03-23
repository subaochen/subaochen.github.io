---
title: Ubuntu下面配置开发环境的一些片段
type: post
categories:
- 高效工作
layout: post
date: 2020-2-16
tags: [linux,vim,ssh,proxy]
status: publish
published: true
comments: true
---

Windows下面配置vim作为开发环境的几个要点记录一下：

今天在Ubuntu下面配置了一下开发环境，把容易忘记的点记录如下：

* 安装v2ray可以参考这里：https://www.imcaviare.com/2018-12-18-1/，配置文件最好在windows下生成，复制过去就可以了。

* 设置ssh走socks代理的方法是在.ssh下面增加一个config文件如下：

  ```
  ProxyCommand nc -x 127.0.0.1:1080 %h %p
  
  Host github.com
    User git
    Port 22
    Hostname github.com
    # 注意修改路径为你的路径
    IdentityFile "/home/subaochen/.ssh/id_rsa"
    TCPKeepAlive yes
  
  Host ssh.github.com
    User git
    Port 443
    Hostname ssh.github.com
    # 注意修改路径为你的路径
    IdentityFile "/home/subaochen/.ssh/id_rsa"
    TCPKeepAlive yes
  
  ```

  

* ssh走socks代理后，安装vim的YCM就轻松多了，安装过程可以参考：https://zhuanlan.zhihu.com/p/35626421。在执行./install.py --all的时候，可能需要安装go语言，也可能因为墙的原因，go的一些依赖无法下载，那么参考[这里](https://blog.csdn.net/u013397318/article/details/80937583)一个一个clone到src/golang.org/x目录下即可。

* 安装anaconda，安装完成修改.condarc，增加清华源：

  ```
  
  channels:
    - defaults
  show_channel_urls: true
  channel_alias: https://mirrors.tuna.tsinghua.edu.cn/anaconda
  default_channels:
    - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
    - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
    - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  custom_channels:
    conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
    msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
    bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
    menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
    pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
    simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  
  ```

  