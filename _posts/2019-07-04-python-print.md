---
title: Python的print的用法
type: post
categories:
- python
layout: post
date: 2019-07-04
tags: [python,print,format]
status: publish
published: true
use_math: false
comments: true
---

## 最简单和自然的输出方式


```python
print("hello, world!")
print("hello," + "world!")
a = 100
print(a)
# 这句会报错
#print("a=" + a)
print("a=" + str(a))
l = ("c","java","python")
print(l)
```

    hello, world!
    hello,world!
    100
    a=100
    ('c', 'java', 'python')


## 使用format格式化输出


```python
a = 100
b = 3.14159
print("a={},b={}".format(a,b))
# 使用和C语言差不多的字符串格式化定义，但是用{}包围格式化字符串，且使用:开头
print("a={:4d},b={:.2f}".format(a,b))
print("{0},{1},{0}".format("hello","world"))
print("name={name},age={age}".format(name="zhangsan",age="22"))
name="zhangsan"
age=22
# format的变种应用，更方便
print(f"name={name},age={age}")
```

    a=100,b=3.14159
    a= 100,b=3.14
    hello,world,hello
    name=zhangsan,age=22
    name=zhangsan,age=22


## 使用%格式化输出
这个网上的资料一大堆，因为已经被format取代，不再赘述。
