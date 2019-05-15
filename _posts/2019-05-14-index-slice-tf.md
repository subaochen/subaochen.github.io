---
title: tensorflow中的索引和切片操作
type: post
categories:
- deeplearning
layout: post
date: 2019-05-14
tags: [numpy, 切片]
status: publish
published: true
comments: true

---
tensorflow中的索引和切片操作支持两种风格，分别简述和演示如下。


```python
import tensorflow as tf
tf.enable_eager_execution()
```

# 基本索引

array[idx][idx][idx]方式的索引，即在方括号中给出张量（多维数组）每一维的索引。


```python
a = tf.ones([1,5,5,3])
a[0]
```




    <tf.Tensor: id=25, shape=(5, 5, 3), dtype=float32, numpy=
    array([[[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]],
    
           [[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]],
    
           [[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]],
    
           [[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]],
    
           [[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]]], dtype=float32)>




```python
a[0][0]
```




    <tf.Tensor: id=34, shape=(5, 3), dtype=float32, numpy=
    array([[1., 1., 1.],
           [1., 1., 1.],
           [1., 1., 1.],
           [1., 1., 1.],
           [1., 1., 1.]], dtype=float32)>




```python
a[0][0][0]
```




    <tf.Tensor: id=47, shape=(3,), dtype=float32, numpy=array([1., 1., 1.], dtype=float32)>




```python
a[0][0][0][0]
```




    <tf.Tensor: id=64, shape=(), dtype=float32, numpy=1.0>



# numpy风格的索引

在一个方括号中列出所有的索引，形如arr[n,m,k]的方式，比基本索引方式简洁直观。


```python
# 模拟创建4张28x28的RGB三通道彩色图片
a = tf.random.normal([4,28,28,3])
# 第0张图片
a[0].shape
```




    TensorShape([Dimension(28), Dimension(28), Dimension(3)])




```python
# 第0张图片第3行
a[0,3].shape
```




    TensorShape([Dimension(28), Dimension(3)])




```python
# 第0张图片第3行第3列的RGB值向量
a[0,3,3].shape
```




    TensorShape([Dimension(3)])




```python
# 第0张图片第3行第3列的RGB中的R的值
a[0,3,3,0].shape
```




    TensorShape([])



# 切片操作

start:end:step方式的切片操作很方便，在每一个维度上都可以切片，如果省略则表示列出该维度上的全部元素。


```python
b = tf.range(10)
b
```




    <tf.Tensor: id=146, shape=(10,), dtype=int32, numpy=array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int32)>




```python
b[:-1]
```




    <tf.Tensor: id=151, shape=(9,), dtype=int32, numpy=array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=int32)>




```python
b[-1:]
```




    <tf.Tensor: id=156, shape=(1,), dtype=int32, numpy=array([9], dtype=int32)>




```python
b[::2]
```




    <tf.Tensor: id=161, shape=(5,), dtype=int32, numpy=array([0, 2, 4, 6, 8], dtype=int32)>




```python
b[::-1]
```




    <tf.Tensor: id=166, shape=(10,), dtype=int32, numpy=array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=int32)>




```python
print(a.shape)
print(a[0].shape)
a[0,:,:,:].shape # 在维度上的切片
```

    (4, 28, 28, 3)
    (28, 28, 3)





    TensorShape([Dimension(28), Dimension(28), Dimension(3)])




```python
a[0,1,:,:].shape
```




    TensorShape([Dimension(28), Dimension(3)])


