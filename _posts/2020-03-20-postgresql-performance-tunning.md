---
title: PostgreSQL性能调整
type: post
categories:
- database
layout: post
date: 2020-03-20
tags: [postgresql,performance]
status: publish
published: true
comments: true
---

下面的配置参数影响或者帮助调整postgresql的性能（重点参考：https://wiki.postgresql.org/wiki/Tuning_Your_PostgreSQL_Server）：

* log_temp_file：最好打开这个选项，当数据库使用磁盘的临时文件进行排序时（即无法在内存中完成排序操作）时，这个选项会记录下来创建磁盘文件的动作，可以据此决定是否则调整内存相关配置选项。
* work_mem：排序操作能够使用的内存。需要注意的是，多用户、多表联合查询很消耗很多倍的work_mem内存，比如work_mem=50M，并发30个用户查询则总内存使用为1.5G。如果涉及到多表联合查询，则使用的内存更多，因此要权衡设置。默认值为4M，可以联合log_temp_file观察是否需要增大work_mem。比如当log_temp_file提示总是创建8M左右的临时文件时，意味着需要设置work_mem为10M左右比较合适。这个设置和具体的业务系统关系比较大。当然，要检查一下log文件，看一下是什么查询导致了临时文件？也许是忘记设置索引导致的，如果能用索引减少内存使用当然不需要动用work_mem设置。
* shared_buffers：这是数据库缓存数据的地方，因此大些有好处。这是一个一次性固定申请的内存，数据库启动的时候就会申请。
* effective_cache_size：数据库在规划查询的时候会参考这个设置来决定对内存的使用方式，因此合理的设置非常重要，通常设置为可使用内存的一半设置更多，可以将top命令的free+cached作为可使用内存。

