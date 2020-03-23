---
title: windows下面配置vim
type: post
categories:
- 高效工作
layout: post
date: 2020-2-16
tags: [vim,youcompleteme]
status: publish
published: true
comments: true
---

Windows下面配置vim作为开发环境的几个要点记录一下：

* 建议安装32版本的vim，因为有人已经编译了32位版本的youcompleteme插件可以直接用：https://www.zhihu.com/question/25437050。
* 需要安装32位版本的python 3.8（配合上面预编译的32位youcompleteme插件）
* 安装ultisnipets插件，不要安装snipmate插件，否则和YCM的快捷键冲突很难解决。
* 注意维护自己添加的snippets内容，重装的时候直接复制进去即可。

