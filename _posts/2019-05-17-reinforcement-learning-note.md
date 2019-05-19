---

title: 强化学习笔记
type: post
categories:
- deeplearning
layout: post
date: 2019-05-17
tags: [深度学习, 强化学习]
status: publish
published: true
comments: true
---

# 强化学习的几个关键词

下面是强化学习各个环节的“角色”，在程序中一般也是这样命名相关变量的：

* **agent**，智能体，游戏中的玩家。
* **env**，环境，或者系统环境（System Environment）。
* **observation**，观察值（从agent的角度观察env获得的env的变化），又叫环境的状态（反映了env的变化过程）。从设计角度看，环境的状态不会全部被agent观测到，但是在强化学习中，一般是从agent的角度看待问题，即agent能够观测到的环境的状态就是observation，一般的等同于env的状态（status）。
* **action**，当agent观察到env的状态发生变化时所采取的行动。
* **reward**，环境对agent的action的反馈或者回报（奖励）。

# 强化学习适合解决什么样的问题？
强化学习天然带有“序列”的成分(s~1, a~1, s~2, a~2,..., s~k, a~k)，因此更适合解决序列（时间等）等相关的问题，比如语言、语音等？有待求证。

# 强化学习中的reward function

其实就是cost function在强化学习中的叫法而已，只不过符号相反，即：
$$
reward function = -cost function
$$

# 参考教程（资料）

* tensorflow2 深度强化学习指南，这是我第一篇阅读的深度强化学习资料，录入了程序，试着运行了一下，发现一脸懵逼：<http://inoryy.com/post/tensorflow2-deep-reinforcement-learning/?spm=a2c4e.11153940.blogcont688842.16.54633e98bex87Y>
* 莫烦python的强化学习教程，还不错，简明扼要：<https://study.163.com/course/courseMain.htm?courseId=1003598006>
* 伯克利的强化学习课程视频，非常精彩，不知道完整的课程视频是否有放出？<https://www.bilibili.com/video/av39816961?p=1>
* 据说这个也挺好，待验证：<https://v.youku.com/v_show/id_XMjcwMDQyOTcxMg==.html?&f=49376145>，知乎有相应专栏：<https://zhuanlan.zhihu.com/p/25498081>