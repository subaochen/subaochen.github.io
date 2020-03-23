---
title: PostgreSQL的慢查询分析
type: post
categories:
- database
layout: post
date: 2020-03-20
tags: [postgresql,slow query]
status: publish
published: true
comments: true
---

这篇讲的非常好：https://www.cybertec-postgresql.com/en/3-ways-to-detect-slow-queries-in-postgresql/，概况起来有三种方法：

* 打开log_min_duration_statement，然后检查log文件发现慢查询，适合于寻找单个的慢查询。
* 借助auto_explain扩展，然后检查log文件检查慢查询的explain，适合于发现不稳定的explain，即时快时慢的查询。
* 借助pg_stat_statements扩展，很强大，适合于发现慢查询的规律。

