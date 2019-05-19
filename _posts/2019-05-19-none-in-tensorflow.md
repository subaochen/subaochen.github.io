---
title: tensorflow中的None
type: post
categories:
- deeplearning
layout: post
date: 2019-05-17
tags: [tensorflow, None]
status: publish
published: true
comments: true
---

在设置tensor的shape时，可以通过None告诉tensorflow，在这个维度上可以接受任意长度的数据，意即，可以是1个，2个，...，N个。

但是，应该只在一个维度上使用None，通常用在第一个维度上，一般的用来表示batch的大小，如果是None，即表示接受任意大小的batch_size。比如在RNN生成文本的例子中，我们通过训练得到了一个模型，那么在预测时需要重构模型，将输入的shape修改为(None,65)，就可以使用任意长度的输入字符串来开始预测流程了。

# 初始化实验环境


```python
import tensorflow as tf
```


```python
def tf_reset():
    try:
        sess.close()
    except:
        pass
    tf.reset_default_graph()
    return tf.Session()
```

# shape为0的情形
shape为()意为标量。


```python
sess = tf_reset()
a = tf.placeholder(tf.float32, shape=(),name="a_placeholder")
b = tf.placeholder(tf.float32, shape=(),name="b_placeholder")
c = a + b
print(sess.run(c, feed_dict={a:1.,b:2.}))
```

    3.0


# 一维张量的情形
在数学上，一维向量通常是指列向量，但是在tensorflow中，一维向量（张量）是指行向量，这一点要注意一下。

## 长度为1的一维向量


```python
sess = tf_reset()
a = tf.placeholder(tf.float32, shape=(1),name="a_placeholder")
b = tf.placeholder(tf.float32, shape=(1),name="b_placeholder")
c = a + b
print(sess.run(c, feed_dict={a:[1.],b:[2.]}))
```

    [3.]


## 长度为2的一维向量


```python
sess = tf_reset()
a = tf.placeholder(tf.float32, shape=(2),name="a_placeholder")
b = tf.placeholder(tf.float32, shape=(2),name="b_placeholder")
c = a + b
print(sess.run(c, feed_dict={a:[1.,2.],b:[3.,4.]}))
```

    [4. 6.]


## 使用None接收任意长度的一维向量


```python
sess = tf_reset()
a = tf.placeholder(tf.float32, shape=(None),name="a_placeholder")
b = tf.placeholder(tf.float32, shape=(None),name="b_placeholder")
c = a + b
print(sess.run(c, feed_dict={a:[1., 2., 3.],b:[4., 5., 6.]}))
```

    [5. 7. 9.]


# 多维向量中的None


```python
sess = tf_reset()
a = tf.placeholder(tf.float32, shape=(None,2),name="a_placeholder")
b = tf.placeholder(tf.float32, shape=(None,2),name="b_placeholder")
c = a + b
print(sess.run(c, feed_dict={a:[[1., 2.],[3., 4.]],b:[[5., 6.],[7., 8.]]}))
```

    [[ 6.  8.]
     [10. 12.]]

