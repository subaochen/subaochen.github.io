---
title: ACTION_VIEW的过滤规则
type: post
categories:
- android
layout: post
date: 2019-09-30
tags: [Intent filter]
status: publish
published: true
comments: true
---

不知道从什么时候开始，Android的intent filter中对VIEW的过滤规则发生了变化，也就是说，如果你自己设计了一个浏览器，则必须在manifest文件中如下声明Activity才会在选项中列出你自己的浏览器：

```xml
            <intent-filter>
                <action android:name="android.intent.action.VIEW"/>
                <category android:name="android.intent.category.DEFAULT"/>
                <category android:name="android.intent.category.BROWSABLE"/>
                <data android:scheme="http"/>
                <data android:scheme="https"/>
            </intent-filter>
```

https://developer.android.com/reference/android/content/Intent#CATEGORY_BROWSABLE 的解释是：

> Activities that can be safely invoked from a browser must support this category. For example, if the user is viewing a web page or an e-mail and clicks on a link in the text, the Intent generated execute that link will require the BROWSABLE category, so that only activities supporting this category will be considered as possible actions. By supporting this category, you are promising that there is nothing damaging (without user intervention) that can happen by invoking any matching Intent.