---
title: pandas学习笔记
type: post
categories:
- python
layout: post
date: 2019-05-19
tags: [python,pandas]
status: publish
published: true
comments: true


---

pandas是基于numpy的数据处理包，是python界的巨无霸，其手册超过2000页，参见下图，大家感受一下：

![pandas manual](/images/python/pandas-manual.png)

其中的getting started部分也有接近200页，即使“10 Minutes to pandas”也有近30页的篇幅！不过，pandas的这个巨无霸手册组织的比较条理，几乎能够解决你的一切问题，应该作为案头的必备工具之一。

建议至少将getting started部分能够通读一遍，了解pandas的强大数据分析方法，重要的功能动手练习一下。下面的内容是本人的练习和其中的一些感悟，不作为系统的教程。pandas如此强大，因此本人也会不断的阅读、练习以加深对pandas的理解，这里也就会有不断的更新。


# pandas的数据结构及其创建
如果把numpy比作列表，那么pandas就是字典。pandas的两个主要数据结构是Series和DataFrame，Series由索引和数据两部分组成，DataFrame是由多个Series组成的，本质上是一张二维表格。

## Series


```python
import numpy as np
import pandas as pd
```


```python
data = [1,2,3,np.nan,55]
serial = pd.Series(data)
print(serial)
```

    0     1.0
    1     2.0
    2     3.0
    3     NaN
    4    55.0
    dtype: float64


可见，如果没有为Series指定索引，pandas会自动创建从0开始的整数索引。下面为Series指定索引：


```python
dates = pd.date_range('20190501', periods=5)
s = pd.Series(data, dates) # 注意参数的顺序，或者这样调用:pd.Series(data=data, index=dates)
print(s)
print(s.index)
print(s.values)
```

    2019-05-01     1.0
    2019-05-02     2.0
    2019-05-03     3.0
    2019-05-04     NaN
    2019-05-05    55.0
    Freq: D, dtype: float64
    DatetimeIndex(['2019-05-01', '2019-05-02', '2019-05-03', '2019-05-04',
                   '2019-05-05'],
                  dtype='datetime64[ns]', freq='D')
    [ 1.  2.  3. nan 55.]


Series的属性具体参见Pandas API reference中的Series部分。

## DataFrame
DataFrame中的index是指行的索引，column是列的索引。

直接借用上面创建的dates作为DateFrame的索引，创建一个DataFrame：


```python
df = pd.DataFrame(index=dates, data=np.random.randn(20).reshape((5,4)), columns=list('ABCD'))
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-05-01</th>
      <td>-1.624347</td>
      <td>-0.528796</td>
      <td>-0.074790</td>
      <td>-1.623435</td>
    </tr>
    <tr>
      <th>2019-05-02</th>
      <td>-0.869758</td>
      <td>0.564534</td>
      <td>-1.537493</td>
      <td>0.158298</td>
    </tr>
    <tr>
      <th>2019-05-03</th>
      <td>0.962677</td>
      <td>0.099774</td>
      <td>-0.804824</td>
      <td>1.613789</td>
    </tr>
    <tr>
      <th>2019-05-04</th>
      <td>0.934002</td>
      <td>0.806180</td>
      <td>-0.141207</td>
      <td>2.509175</td>
    </tr>
    <tr>
      <th>2019-05-05</th>
      <td>-2.036735</td>
      <td>0.668513</td>
      <td>0.773942</td>
      <td>2.131727</td>
    </tr>
  </tbody>
</table>
</div>



查看DataFrame的索引：


```python
df.index
```




    DatetimeIndex(['2019-05-01', '2019-05-02', '2019-05-03', '2019-05-04',
                   '2019-05-05'],
                  dtype='datetime64[ns]', freq='D')



查看DataFrame的columns：


```python
df.columns
```




    Index(['A', 'B', 'C', 'D'], dtype='object')



查看DataFrames的数据：


```python
df.values
```




    array([[-1.62434694, -0.52879577, -0.07479036, -1.62343509],
           [-0.86975781,  0.56453382, -1.53749287,  0.1582984 ],
           [ 0.96267722,  0.09977408, -0.80482377,  1.61378861],
           [ 0.93400177,  0.80617994, -0.141207  ,  2.50917471],
           [-2.03673517,  0.66851319,  0.77394167,  2.1317268 ]])



# DataFrame的数据选取
从DataFrame中选取数据主要有三种方法，分别通过[],loc,iloc属性实现。因此，数据选取的操作都是通过[]，注意和函数调用的()区分开来：数据选取不是函数调用。

## 通过[]选取数据
[]是pandas选取数据的操作符，注意和列表[]区分开来。


```python
df['A']
```




    2019-05-01   -1.624347
    2019-05-02   -0.869758
    2019-05-03    0.962677
    2019-05-04    0.934002
    2019-05-05   -2.036735
    Freq: D, Name: A, dtype: float64




```python
df[['A','D']] # 注意只是写df['A','D']是不行的，因为外层的[]表示一个选取操作，里层的[]表示选取哪些列的一个列表
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-05-01</th>
      <td>-1.624347</td>
      <td>-1.623435</td>
    </tr>
    <tr>
      <th>2019-05-02</th>
      <td>-0.869758</td>
      <td>0.158298</td>
    </tr>
    <tr>
      <th>2019-05-03</th>
      <td>0.962677</td>
      <td>1.613789</td>
    </tr>
    <tr>
      <th>2019-05-04</th>
      <td>0.934002</td>
      <td>2.509175</td>
    </tr>
    <tr>
      <th>2019-05-05</th>
      <td>-2.036735</td>
      <td>2.131727</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[0:3]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-05-01</th>
      <td>-1.624347</td>
      <td>-0.528796</td>
      <td>-0.074790</td>
      <td>-1.623435</td>
    </tr>
    <tr>
      <th>2019-05-02</th>
      <td>-0.869758</td>
      <td>0.564534</td>
      <td>-1.537493</td>
      <td>0.158298</td>
    </tr>
    <tr>
      <th>2019-05-03</th>
      <td>0.962677</td>
      <td>0.099774</td>
      <td>-0.804824</td>
      <td>1.613789</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['20190502':'20190504']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-05-02</th>
      <td>-0.869758</td>
      <td>0.564534</td>
      <td>-1.537493</td>
      <td>0.158298</td>
    </tr>
    <tr>
      <th>2019-05-03</th>
      <td>0.962677</td>
      <td>0.099774</td>
      <td>-0.804824</td>
      <td>1.613789</td>
    </tr>
    <tr>
      <th>2019-05-04</th>
      <td>0.934002</td>
      <td>0.806180</td>
      <td>-0.141207</td>
      <td>2.509175</td>
    </tr>
  </tbody>
</table>
</div>



## 通过loc选取数据（给定label）
loc要求给出具体的行或列的label或者其范围选取数据。


```python
df.loc[dates[0]]
```




    A   -1.624347
    B   -0.528796
    C   -0.074790
    D   -1.623435
    Name: 2019-05-01 00:00:00, dtype: float64




```python
df.loc[:,['A','C']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-05-01</th>
      <td>-1.624347</td>
      <td>-0.074790</td>
    </tr>
    <tr>
      <th>2019-05-02</th>
      <td>-0.869758</td>
      <td>-1.537493</td>
    </tr>
    <tr>
      <th>2019-05-03</th>
      <td>0.962677</td>
      <td>-0.804824</td>
    </tr>
    <tr>
      <th>2019-05-04</th>
      <td>0.934002</td>
      <td>-0.141207</td>
    </tr>
    <tr>
      <th>2019-05-05</th>
      <td>-2.036735</td>
      <td>0.773942</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc['20190501',['A']]
```




    A   -1.624347
    Name: 2019-05-01 00:00:00, dtype: float64




```python
df.at[dates[0],'A']
```




    -1.6243469408972495



## 通过iloc选取数据（给定position）


```python
print(df)
df.iloc[1,1]
```

                       A         B         C         D
    2019-05-01 -1.624347 -0.528796 -0.074790 -1.623435
    2019-05-02 -0.869758  0.564534 -1.537493  0.158298
    2019-05-03  0.962677  0.099774 -0.804824  1.613789
    2019-05-04  0.934002  0.806180 -0.141207  2.509175
    2019-05-05 -2.036735  0.668513  0.773942  2.131727





    0.564533818704376




```python
df.iloc[3]
```




    A    0.934002
    B    0.806180
    C   -0.141207
    D    2.509175
    Name: 2019-05-04 00:00:00, dtype: float64




```python
df.iloc[:,1]
```




    2019-05-01   -0.528796
    2019-05-02    0.564534
    2019-05-03    0.099774
    2019-05-04    0.806180
    2019-05-05    0.668513
    Freq: D, Name: B, dtype: float64




```python
df.iloc[:,1:3]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-05-01</th>
      <td>-0.528796</td>
      <td>-0.074790</td>
    </tr>
    <tr>
      <th>2019-05-02</th>
      <td>0.564534</td>
      <td>-1.537493</td>
    </tr>
    <tr>
      <th>2019-05-03</th>
      <td>0.099774</td>
      <td>-0.804824</td>
    </tr>
    <tr>
      <th>2019-05-04</th>
      <td>0.806180</td>
      <td>-0.141207</td>
    </tr>
    <tr>
      <th>2019-05-05</th>
      <td>0.668513</td>
      <td>0.773942</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.iat[0,0]
```




    -1.6243469408972495



# 关于索引的操作

## 返回Series中最大/最小数据的column名称


```python
s = df.loc[dates[0]]
s.idxmax()
```




    'C'




```python
s.idxmin()
```




    'A'



## 返回Series中最大/最小数据的column位置
这里注意，直接调用Series.argmax等同于idxmax且已经depricated。


```python
print(s)
print(s.values.argmax())
```

    A   -1.624347
    B   -0.528796
    C   -0.074790
    D   -1.623435
    Name: 2019-05-01 00:00:00, dtype: float64
    2



```python
s.values.argmin()
```




    0



# DataFrame上的逻辑操作
可以在整个或者部分DataFrame执行all或者any逻辑操作。


```python
(df == 0).all()
```




    A    False
    B    False
    C    False
    D    False
    dtype: bool




```python
(df.iloc[0,:] > 0).all()
```




    False


