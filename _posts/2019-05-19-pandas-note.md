---
title: pandas学习笔记
type: post
categories:
- python
layout: post
date: 2019-05-19
tags: [python, pandas]
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
      <td>0.458107</td>
      <td>-0.192635</td>
      <td>1.833757</td>
      <td>1.018542</td>
    </tr>
    <tr>
      <th>2019-05-02</th>
      <td>-0.619739</td>
      <td>0.902050</td>
      <td>0.740512</td>
      <td>0.734122</td>
    </tr>
    <tr>
      <th>2019-05-03</th>
      <td>0.281957</td>
      <td>-2.269712</td>
      <td>-1.119312</td>
      <td>-1.225087</td>
    </tr>
    <tr>
      <th>2019-05-04</th>
      <td>-0.269801</td>
      <td>0.527882</td>
      <td>-0.823341</td>
      <td>-0.748891</td>
    </tr>
    <tr>
      <th>2019-05-05</th>
      <td>0.293009</td>
      <td>-0.111322</td>
      <td>-1.520751</td>
      <td>-0.728785</td>
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




    array([[ 0.45810656, -0.19263504,  1.83375697,  1.0185416 ],
           [-0.61973932,  0.9020497 ,  0.740512  ,  0.73412183],
           [ 0.28195677, -2.26971194, -1.11931152, -1.2250866 ],
           [-0.26980061,  0.52788194, -0.82334148, -0.7488908 ],
           [ 0.29300924, -0.11132153, -1.52075088, -0.72878478]])



# DataFrame的数据选取
从DataFrame中选取数据主要有三种方法，分别通过[],loc,iloc属性实现。因此，数据选取的操作都是通过[]，注意和函数调用的()区分开来：数据选取不是函数调用。

## 通过[]选取数据
[]是pandas选取数据的操作符，注意和列表[]区分开来。


```python
df['A']
```




    2019-05-01    0.458107
    2019-05-02   -0.619739
    2019-05-03    0.281957
    2019-05-04   -0.269801
    2019-05-05    0.293009
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
      <td>0.458107</td>
      <td>1.018542</td>
    </tr>
    <tr>
      <th>2019-05-02</th>
      <td>-0.619739</td>
      <td>0.734122</td>
    </tr>
    <tr>
      <th>2019-05-03</th>
      <td>0.281957</td>
      <td>-1.225087</td>
    </tr>
    <tr>
      <th>2019-05-04</th>
      <td>-0.269801</td>
      <td>-0.748891</td>
    </tr>
    <tr>
      <th>2019-05-05</th>
      <td>0.293009</td>
      <td>-0.728785</td>
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
      <td>0.458107</td>
      <td>-0.192635</td>
      <td>1.833757</td>
      <td>1.018542</td>
    </tr>
    <tr>
      <th>2019-05-02</th>
      <td>-0.619739</td>
      <td>0.902050</td>
      <td>0.740512</td>
      <td>0.734122</td>
    </tr>
    <tr>
      <th>2019-05-03</th>
      <td>0.281957</td>
      <td>-2.269712</td>
      <td>-1.119312</td>
      <td>-1.225087</td>
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
      <td>-0.619739</td>
      <td>0.902050</td>
      <td>0.740512</td>
      <td>0.734122</td>
    </tr>
    <tr>
      <th>2019-05-03</th>
      <td>0.281957</td>
      <td>-2.269712</td>
      <td>-1.119312</td>
      <td>-1.225087</td>
    </tr>
    <tr>
      <th>2019-05-04</th>
      <td>-0.269801</td>
      <td>0.527882</td>
      <td>-0.823341</td>
      <td>-0.748891</td>
    </tr>
  </tbody>
</table>
</div>



## 通过loc选取数据（给定label）
loc要求给出具体的行或列的label或者其范围选取数据。


```python
df.loc[dates[0]]
```




    A    0.458107
    B   -0.192635
    C    1.833757
    D    1.018542
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
      <td>0.458107</td>
      <td>1.833757</td>
    </tr>
    <tr>
      <th>2019-05-02</th>
      <td>-0.619739</td>
      <td>0.740512</td>
    </tr>
    <tr>
      <th>2019-05-03</th>
      <td>0.281957</td>
      <td>-1.119312</td>
    </tr>
    <tr>
      <th>2019-05-04</th>
      <td>-0.269801</td>
      <td>-0.823341</td>
    </tr>
    <tr>
      <th>2019-05-05</th>
      <td>0.293009</td>
      <td>-1.520751</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc['20190501',['A']]
```




    A    0.458107
    Name: 2019-05-01 00:00:00, dtype: float64




```python
df.at[dates[0],'A']
```




    0.45810656189864746



## 通过iloc选取数据（给定position）


```python
print(df)
df.iloc[1,1]
```

                       A         B         C         D
    2019-05-01  0.458107 -0.192635  1.833757  1.018542
    2019-05-02 -0.619739  0.902050  0.740512  0.734122
    2019-05-03  0.281957 -2.269712 -1.119312 -1.225087
    2019-05-04 -0.269801  0.527882 -0.823341 -0.748891
    2019-05-05  0.293009 -0.111322 -1.520751 -0.728785





    0.9020496997689477




```python
df.iloc[3]
```




    A   -0.269801
    B    0.527882
    C   -0.823341
    D   -0.748891
    Name: 2019-05-04 00:00:00, dtype: float64




```python
df.iloc[:,1]
```




    2019-05-01   -0.192635
    2019-05-02    0.902050
    2019-05-03   -2.269712
    2019-05-04    0.527882
    2019-05-05   -0.111322
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
      <td>-0.192635</td>
      <td>1.833757</td>
    </tr>
    <tr>
      <th>2019-05-02</th>
      <td>0.902050</td>
      <td>0.740512</td>
    </tr>
    <tr>
      <th>2019-05-03</th>
      <td>-2.269712</td>
      <td>-1.119312</td>
    </tr>
    <tr>
      <th>2019-05-04</th>
      <td>0.527882</td>
      <td>-0.823341</td>
    </tr>
    <tr>
      <th>2019-05-05</th>
      <td>-0.111322</td>
      <td>-1.520751</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.iat[0,0]
```




    0.45810656189864746



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




    'B'



## 返回Series中最大/最小数据的column位置
这里注意，直接调用Series.argmax等同于idxmax且已经depricated。


```python
print(s)
print(s.values.argmax())
```

    A    0.458107
    B   -0.192635
    C    1.833757
    D    1.018542
    Name: 2019-05-01 00:00:00, dtype: float64
    2



```python
s.values.argmin()
```




    1


