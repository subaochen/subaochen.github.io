---
title: tuple的构造方法：单个元素的情况
type: post
categories:
- python
layout: post
date: 2019-07-29
tags: [python,tuple]
status: publish
published: true
comments: true
---

在创建python的tuple时，众所周知直接使用小括号即可，比如：


```python
a = (1,2,3)
a
```


    (1, 2, 3)

但是，如果tuple中只有一个元素呢？比如：


```python
b = (1)
b
```


    1

从上面的运行结果看，显然输出的b不是一个tuple，只是一个整数1。原因在于，小括号在这里被解释为算数运算表达式中小括号：用来表示运算等级。为了正确创建tuple，可以有两种方式：


```python
b = (1,)
b
```


    (1,)


```python
b = tuple([1])
b
```


    (1,)

如果不小心忘记了这条规则，python有时候会报出莫名其妙的错误信息，比如在使用psycopg2连接数据库时，有如下的sql：

```sql
sql = "select id from tab where code=%s"
```

在具体执行时如果这样写：

```python
cur.execute(sql, (code))
```

其中的code是一个整数类型的代码编号，则会报错：

```
TypeError: not all arguments converted during string formatting
```

是不是有点莫名其妙？其实，只要把上面的(code)修改为(code,)或者tuple([code])就可以了。也就是说，execute要求第二个参数是tuple类型的，如果只是给出整数或者字符串，就会报告上面的错误信息。