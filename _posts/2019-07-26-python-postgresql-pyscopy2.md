使用pyscopy2连接postgresql数据库服务器的步骤，简单记录一下：

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

# 连接数据库

