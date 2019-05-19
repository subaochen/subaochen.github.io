---
title: jupyter使用技巧
type: post
categories:
- python
layout: post
date: 2019-05-15
tags: [jupyter]
status: publish
published: true
comments: true

---

这里汇集一些jupyter的使用技巧。

# 安装新的jupyter主题

默认的jupyter主题背景太亮，解决这个问题一般有两个办法：

## 使用JupyterLab
jupyterlab默认自带dark主题。jupyterlab的安装方法：

> pip install jupyterlab

注意，启动jupyterlab的命令是`jupyter-lab`，而不是`jupyterlab`，不知道
jupyterlab的作者为何这样安排，有点混乱。

## 安装jupyterthemes包

> pip install jupyterthemes

> jt -t chesterish

然后就可以在jupyter界面看到舒服的深色主题了。

# 常用快捷键

* `esc`：离开当前的模式，这和vi非常相似。实际上，jupyer的很多快捷键似乎就是从vi借鉴过来的。
* `a`: 在上方（above）插入cell。
* `b`: 在下方（below）插入cell。
* `dd`：删除当前cell。
* `m`: 把cell格式变为markdown
* `y`：把cell格式变为代码
* `j`/`k`: 焦点移动到上面/下面的cell

实际上，掌握了`a/b/m/y/j/k`这几个快捷键，几乎就可以抛开鼠标，愉快的玩耍jupyter了。

# 运行shell命令

通过在命令前使用`!`可直接执行shell命令。
