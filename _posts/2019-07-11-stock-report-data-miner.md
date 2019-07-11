---
title: A股年报数据提取小工具
type: post
categories:
- python
layout: post
date: 2019-07-11
tags: [python,pandas,financial,stock]
status: publish
published: true
comments: true
---

孩子的老师布置了个“作业”，要求查出一堆（大概50多家）A股上市公司2013年的研发费用。作为一名光荣的程序猿，自然想到编个小工具来对付，说干就干！

首先对比了新浪财经、东方财富和网易财经的页面数据，发现只有网易提供了非常方便的财务数据下载，为网易点赞！在编写程序的时候，本来想直接利用pandas提供的excel/csv读写功能，无奈发现pandas在写入excel/csv时总是要带个“帽”（表头数据），怎么也去不掉，于是转而采用了自己控制csv格式的“笨”方法。也许是对强大的pandas没有吃透，还望高手指点！

程序写完了才发现，A股从2018年才要求披露研发费用数据，2013年哪来的研发费用？不过既然写了程序，无论需要哪一年的什么数据，都只在弹指之间了。

附录程序如下：

```python
# -*- coding: utf-8 -*-

import urllib3
import pandas as pd
import csv
import os

# if DOWNLOAD = False，don't download report
DOWNLOAD=True

def download_report(code):
    """
    下载报表（利润表年表，如果下载其他报表，需要修改url

    Arguments:
        code: 股票代码
    """
    # 利润表(年表)
    url = f"http://quotes.money.163.com/service/lrb_{code}.html?type=year"
    base_dir='data'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    report_file = f'{base_dir}/lrb_{code}.csv'
    http = urllib3.PoolManager()

    response = http.request('GET', url)
    with open(report_file, 'wb') as f:
        # 163网页的编码是gbk，需要转成utf-8
        f.write(response.data.decode('gbk').encode('utf-8'))


def analyse_report(code, rows, cols):
    """
    分析报表
    Arguments:
        code:股票代码
        rows:读取报表的哪些行，接受列表
        cols:读取报表的哪些列，接受列表

    Return:
        获取的数据
    """
    filename = f'data/lrb_{code}.csv'
    df = pd.read_csv(filename)
    data = df.iloc[rows, cols]
    data.insert(0, '股票代码', code)
    return data

if __name__ == '__main__':
    # 测试股票代码集合
    codes = ['603730', '603757']
    summary_data = []
    for code in codes:
        if DOWNLOAD == True:
            download_report(code)
        # 第一列（2018年数据），第12行（研发费用）
        data = analyse_report(code, [12], [1])
        summary_data.append(data.values.tolist())

    with open("summary.csv", "w+") as csvfile:
        writer = csv.writer(csvfile)
        # 根据情况写入表头
        writer.writerow(["股票代码", "研发费用(万元)"])
        # 写入多行用writerows
        for item in summary_data:
            writer.writerows(item)

```



