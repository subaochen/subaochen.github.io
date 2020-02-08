---
title: 使用vim的snippets方便的创建pgsql的函数
type: post
categories:
- 高效工作
layout: post
date: 2020-2-8
tags: [vim,snippets,postgresql,function]
status: publish
published: true
comments: true
---

最近要写一堆PostgreSQL的函数，每次都复制粘贴函数的定义肯定不胜其烦，好在vim有snippets这个利器，方便了不少。

# snippets插件的安装

在vundle的管理下，snippets插件的安装只需要在.vimrc中添加如下的代码行即可：

```vim
" snippets support
Plugin 'MarcWeber/vim-addon-mw-utils'
Plugin 'tomtom/tlib_vim'
Plugin 'garbas/vim-snipmate'
Plugin 'honza/vim-snippets' "massive common snippets
```

然后执行`:PluginInstall`即可完成snippets插件的安装。

snippets插件默认附带了非常多的snippets文件，这里只需要找到bundle\vim-snippets\snippets这个目录，然后编辑sql.snippets，在最后增加如下的定义：

```
snippet	fun
	--
	DROP FUNCTION IF EXISTS ${1:funname}(${2:params});
	CREATE OR REPLACE FUNCTION $1(
		$2 --
	)
	RETURNS integer AS
	$BODY$
	DECLARE
	BEGIN
		${0}
		return 1;
	END;
	$BODY$
	LANGUAGE plpgsql VOLATILE
	COST 100;
	ALTER FUNCTION $1($2)
	OWNER TO postgres;
```

snippets的语法很简单：`${1}`,`${2}`等表示按Tab键时跳转的位置，数字代表了跳转的顺序。`${0}`有特殊的含义，表示最后一跳。`${1:funname}`表示第一个跳转默认的内容是funname，即如果在第一跳如果不做修改直接再按Tab的话，就采用这个默认值了。

需要注意的是，开头的snippets和fun之间是tab而不是space。

# 插件的实际使用效果

完成了上述的设置，编写一个PostgreSQL的函数就变的有趣和简单了，看视频：

<iframe src="//player.bilibili.com/player.html?aid=87462502&cid=149444317&page=1" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true" width="600"> </iframe>
