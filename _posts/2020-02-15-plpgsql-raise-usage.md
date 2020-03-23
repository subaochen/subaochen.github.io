---
title: 合理使用raise的级别
type: post
categories:
- PostgreSQL
layout: post
date: 2020-2-15
tags: [PostgreSQL,plpgsql,raise]
status: publish
published: true
comments: true
---

使用raise打印信息的时候，需要合理设置raise的级别，过多的输出信息即影响执行的速度，也不便于调试。影响raise信息显示的有两个设置：

* client_min_messages：决定在客户端显示什么级别的raise信息，依次为：DEBUG5`、        `DEBUG4`、`DEBUG3`、`DEBUG2`、        `DEBUG1`、`LOG`、`NOTICE`、        `WARNING`、`ERROR`、`FATAL`和`PANIC，默认值是NOTICE。
* log_min_messages：决定了那些信息写入日志，依次为：DEBUG5`、`DEBUG4`、        `DEBUG3`、`DEBUG2`、`DEBUG1`、        `INFO`、`NOTICE`、`WARNING`、        `ERROR`、`LOG`、`FATAL`和        `PANIC，默认值是WARNING。

可以看出，client_min_messages和log_min_messages的级别设置是不一样的，对比一下便知，以下的原则可能比较实用：

* 尽量避免使用INFO级别，因为在client_min_messages中居然没法这个级别的设置。
* 尽量避免使用LOG级别，因为这会导致大量的信息写入到日志中去，也许这不是你希望的。
* 如果只希望在调试阶段打印到终端上来，则使用DEBUG级别，显然这正是DEBUG级别的真正意义。默认情况下DEBUG级别的信息不会写日志，也不会打印到终端，需要执行命令：`set client_min_messages='debug'`调整raise信息输出级别。在调试完成后，再通过命令`set client_min_messages='notice'`恢复默认的设置，以免影响程序正常执行的效率。任何时候，都可以通过`show client_min_messages`查询当前的设置。
* 如果希望一直在终端显示信息，则使用NOTICE级别是个好主意。一方面，默认的终端输出级别就是NOTICE，无需额外的client_min_messages设置就可以显示NOTICE级别的信息；另一方面，默认情况下，NOTICE级别的信息也不会写入日志。

简言之，**使用debug级别写调试信息，使用notice级别写提示信息**。