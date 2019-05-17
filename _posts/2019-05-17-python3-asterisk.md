---
title: python3中的*用法拾遗
type: post
categories:
- python
layout: post
date: 2019-05-17
tags: [python]
status: publish
published: true
comments: true

---



python3中对*的使用有两个地方需要特别注意一下，分别位于函数的参数和函数的返回值中。

# 函数参数中的*
可以通过*强制此后的参数在实际调用时必须给出参数的名称，以更清楚的表明参数值的含义。


```python
def f(name, *, age=21):
    print(name,",",age)
    
f("zhangsan")
```

    zhangsan , 21



```python
f("zhangsan",21)
```


    ---------------------------------------------------------------------------
    
    TypeError                                 Traceback (most recent call last)
    
    <ipython-input-3-d3f8996c48b8> in <module>
    ----> 1 f("zhangsan",21)


    TypeError: f() takes 1 positional argument but 2 were given



```python
f("zhangsan", age=22)
```

    zhangsan , 22


# 函数返回值中的*
python允许函数返回多个值，在接收这些返回值时，可以通过*简化变量的个数。


```python
def f():
    return "zhangsan",89,95,100,21 # 姓名，语文成绩，数学成绩，英语成绩，年龄
```


```python
name, *scores, age = f()
print(name)
print(scores)
print(age)
```

    zhangsan
    [89, 95, 100]
    21

