---
title: 使用pyscopg2连接PostGreSQL数据库
type: post
categories:
- others
layout: post
date: 2019-07-26
tags: [pyscopg2, postgresql]
status: publish
published: true
comments: true
---

使用pyscopg2连接postgresql数据库服务器的步骤，简单记录一下：

# 准备工作

1. 安装postgresql、pgadmin3

   ```shell
   sudo apt install postgresql pgadmin3
   ```

1. 安装postgresql-server-dev-10。如果只是使用postgresql，开发包也可以不安装，不过下面的psycopy2需要使用postgresql的开发包编译相应的模块，因此需要预先安装：

   ```shell
   sudo apt install postgresql-server-dev-10
   ```

1. 安装psycopy2

   ```shell
   pip install psycopy2
   ```

# 基本用法

```python
import psycopg2
import sys
```

```python
con = psycopg2.connect(database='postgres', user='postgres',
    password='')

with con:
    cur = con.cursor()
    cur.execute('SELECT version()')

    version = cur.fetchone()[0]
    print(version)
```

    PostgreSQL 10.9 (Ubuntu 10.9-0ubuntu0.18.04.1) on x86_64-pc-linux-gnu, compiled by gcc (Ubuntu 7.4.0-1ubuntu1~18.04.1) 7.4.0, 64-bit


# 执行查询


```python
con = psycopg2.connect(database='testdb', user='postgres', password='')

with con:
    cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS cars")
    cur.execute("CREATE TABLE cars(id SERIAL PRIMARY KEY, name VARCHAR(255), price INT)")
    cur.execute("INSERT INTO cars(name, price) VALUES('Audi', 52642)")
    cur.execute("INSERT INTO cars(name, price) VALUES('Mercedes', 57127)")
    cur.execute("INSERT INTO cars(name, price) VALUES('Skoda', 9000)")
    cur.execute("INSERT INTO cars(name, price) VALUES('Volvo', 29000)")
    cur.execute("INSERT INTO cars(name, price) VALUES('Bentley', 350000)")
    cur.execute("INSERT INTO cars(name, price) VALUES('Citroen', 21000)")
    cur.execute("INSERT INTO cars(name, price) VALUES('Hummer', 41400)")
    cur.execute("INSERT INTO cars(name, price) VALUES('Volkswagen', 21600)")
```

# executemany


```python
cars = (
    (1, 'Audi', 52642),
    (2, 'Mercedes', 57127),
    (3, 'Skoda', 9000),
    (4, 'Volvo', 29000),
    (5, 'Bentley', 350000),
    (6, 'Citroen', 21000),
    (7, 'Hummer', 41400),
    (8, 'Volkswagen', 21600)
)

con = psycopg2.connect(database='testdb', user='postgres', password='')

with con:
    cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS cars")
    cur.execute("CREATE TABLE cars(id SERIAL PRIMARY KEY, name VARCHAR(255), price INT)")
    query = "INSERT INTO cars (id, name, price) VALUES (%s, %s, %s)"
    cur.executemany(query, cars)
    con.commit()
```

# fetchall


```python
con = psycopg2.connect(database='testdb', user='postgres',
                    password='s$cret')

with con:
    cur = con.cursor()
    cur.execute("SELECT * FROM cars")
    rows = cur.fetchall()
    for row in rows:
        print(f"{row[0]} {row[1]} {row[2]}")
```

    1 Audi 52642
    2 Mercedes 57127
    3 Skoda 9000
    4 Volvo 29000
    5 Bentley 350000
    6 Citroen 21000
    7 Hummer 41400
    8 Volkswagen 21600


# fetchone


```python
con = psycopg2.connect(database='testdb', user='postgres',
                    password='')

with con:
    cur = con.cursor()
    cur.execute("SELECT * FROM cars")

    while True:
        row = cur.fetchone()
        if row == None:
            break
        print(f"{row[0]} {row[1]} {row[2]}")
```

    1 Audi 52642
    2 Mercedes 57127
    3 Skoda 9000
    4 Volvo 29000
    5 Bentley 350000
    6 Citroen 21000
    7 Hummer 41400
    8 Volkswagen 21600


# 参数化查询

## ANSI C printf format


```python
uId = 1
uPrice = 62300

con = psycopg2.connect(database='testdb', user='postgres',
                    password='')
with con:
    cur = con.cursor()
    cur.execute("UPDATE cars SET price=%s WHERE id=%s", (uPrice, uId))
    print(f"Number of rows updated: {cur.rowcount}")
```

    Number of rows updated: 1


## Python extended format


```python
uid = 3
con = psycopg2.connect(database='testdb', user='postgres',
                    password='')
with con:
    cur = con.cursor()
    cur.execute("SELECT * FROM cars WHERE id=%(id)s", {'id': uid } )
    row = cur.fetchone()
    print(f'{row[0]} {row[1]} {row[2]}')
```

    3 Skoda 9000


# 导入和导出


```python
con = None
fout = None

try:
    con = psycopg2.connect(database='testdb', user='postgres',
                    password='')

    cur = con.cursor()
    fout = open('cars.csv', 'w')
    cur.copy_to(fout, 'cars', sep="|")
    # cur.copy_from

except psycopg2.DatabaseError as e:
    print(f'Error {e}')
    sys.exit(1)
except IOError as e:
    print(f'Error {e}')
    sys.exit(1)
finally:
    if con:
        con.close()
    if fout:
        fout.close()
```

更详细的内容可参考：

* [http://zetcode.com/python/psycopg2/](http://zetcode.com/python/psycopg2/)，大部分代码来自这里。

* [https://github.com/psycopg/psycopg2](https://github.com/psycopg/psycopg2) 的doc部分
* [http://www.postgresqltutorial.com/postgresql-python](http://www.postgresqltutorial.com/postgresql-python/)
* [https://wiki.postgresql.org/wiki/Psycopg2_Tutorial](https://wiki.postgresql.org/wiki/Psycopg2_Tutorial)

