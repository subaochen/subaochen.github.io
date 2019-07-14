---
title:python的yield用法
type: post
categories:
- python
layout: post
date: 2019-07-14
tags: [python,pandas,yield]
status: publish
published: true
comments: true
---

yield可以让一个函数变成“生成器”（generator），每次调用这个函数时通过yield返回一个值，下次调用这个函数时会从yield的下一条语句开始执行，直到遇到yield返回一个值或者return语句终止函数执行。因此，yield配合循环或者迭代过程可以构造简洁直观的代码，比如斐波那契数列可以这样写：


```python
def fabonacci(num_items):
    a,b,n = 0,1,0
    while(n < num_items):
        yield b
        a,b,n = b,(a+b),n+1
```

```python
for n in fabonacci(5):
    print(n)
```

    1
    1
    2
    3
    5


又比如，在处理excel或者csv文件时，可以使用yield来获取指定的内容（感谢[zzliu](https://github.com/zz2liu)的指点，参见[A股年报数据提取小工具](https://subaochen.github.io/python/2019/07/11/stock-report-data-miner/)的评论内容）：


```python
codes = ['603730', '603757']
years = ['2018-12-31', '2017-12-31']
def each_record(codes, years, row='营业收入(万元)'):
    for code in codes:
        filename = f'data/lrb_{code}.csv'
        df = pd.read_csv(filename, index_col=0)
        x = df.loc[row, years]
        yield [code] + list(x)
summary = pd.DataFrame.from_records(each_record(codes, years), columns=['code'] + years)
summary.to_csv('tmp.csv', index=False)        
```

上面这段代码在这里不能直接执行，感兴趣的读者可以参考[stock-report-gen.py](https://github.com/subaochen/financial-tools/blob/master/stock-report-gen.py)。这里解释如下：
each_record方法是一个generator，每次执行获得csv文件指定的数据并通过yield返回，然后通过pandas的DataFrame.from_records方法，巧妙的传入了each_record方法，获得生成汇总csv文件必须的数据。

