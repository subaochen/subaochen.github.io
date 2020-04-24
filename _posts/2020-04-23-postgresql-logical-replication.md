---
title: PostgreSQL的logical replication实战
type: post
categories:
- database
layout: post
date: 2020-04-23
tags: [postgresql, logical replication]
status: publish
published: true
comments: true
---

# 配置实录

PostgreSQL的logical replication配置方法实录，大致分为以下的几个步骤：

1. 在两台服务器上面（假设分别较做db_master和db_slaver）都修改postgresql.con如下：

   `wal_level = logical`

2. 在db_master创建合适的role用于logical replication，比如：

   `create role replica with replication login password 'my_password'`
   
3. 在db_master和db_slave创建数据库和需要同步的数据表（略），假设需要同步的数据库叫test，需要同步的数据表叫test。

4. 在db_master授权replica role对test的访问：

   `grant all privileges on database test to replica;grant all privileges on all tables in schema public to replica;`

5. 在db_master创建publication：

   `create publication mypub;alter publication mypub add table test;`

6. 在db_slaver创建subscription：

   `create subscription mysub connection 'host=master_ip port=master_port password=my_password user=replica dbname=test' publication mypub;`

# 问题排除

1. 在db_master可以通过以下命令查看所创建的publication：

   ```
   select * from pg_publication;
   select * from pg_publication_tables;
   select * from pg_stat_publication;
   ```
   
1. 在db_slaver数据库上，需要同步的数据表不能和db_master的数据表有冲突，这基本上意味着初始的时候，db_slaver上需要同步的数据表最好清空（truncate）一下。

1. 如果存在多个db_master，在每个db_slaver上面执行`create subscription`命令创建一个订阅。自然，每个db_slaver的subscription名称应该不同，这样db_master才知道要把数据发送给谁。

1. 很遗憾的是，`create subscription`命令不能通过psql -c的方式来执行，否则会报错：`...cannot run inside a transaction block psql`，导致如果需要创建很多的db_slaver的时候比较麻烦，需要一台一台的在db_slaver上面执行创建subscription的命令。也许我对psql的理解不到位，也许有方法可以通过脚本的方式执行`create subscription`命令，希望知道的朋友多多指点，先谢过了！