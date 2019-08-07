---
title: Android Studio在国内访问dl.google.com的方法
type: post
categories:
- android
layout: post
date: 2019-08-08
tags: [android]
status: publish
published: true
comments: true
---

Android Studio默认访问dl.google.com下载镜像等，但是在国内经常莫名其妙就访问不了dl.google.com，全凭运气。感谢aliyun提供了镜像服务，只需要将build.properties中的下列两行：

```
google()
jcenter()
```

替换为：

```
    maven {
        url 'https://maven.aliyun.com/repository/google/'
    }
    maven {
        url 'https://maven.aliyun.com/repository/jcenter/'
    }
```

注意，可能有两处需要替换，我这边替换后的build.properties文件如下：

```
// Top-level build file where you can add configuration options common to all sub-projects/modules.

buildscript {
    repositories {
       // google()
       // jcenter()
        maven {
            url 'https://maven.aliyun.com/repository/google/'
        }
        maven {
            url 'https://maven.aliyun.com/repository/jcenter/'
        }
        
    }
    dependencies {
        classpath 'com.android.tools.build:gradle:3.4.2'
        
        // NOTE: Do not place your application dependencies here; they belong
        // in the individual module build.gradle files
    }
}

allprojects {
    repositories {
        //google()
        //jcenter()
        maven {
            url 'https://maven.aliyun.com/repository/google/'
        }
        maven {
            url 'https://maven.aliyun.com/repository/jcenter/'
        }

        
    }
}

task clean(type: Delete) {
    delete rootProject.buildDir
}
```

另外，如果设置了代理，需要取消代理设置，尤其是要检查`$HOME/.gradle/gradle.properties`文件中是否存在代理设置，一定要记得去掉，否则访问不了aliyun的资源。