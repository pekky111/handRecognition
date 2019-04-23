# handRecognition
这是一个手势识别应用程序，主要使用CNN，Tensorflow和OpenCV。
数据集在源码文件夹中
模型和录屏在百度网盘链接中，链接：https://pan.baidu.com/s/10yPpFVxoPXyD-KS3Wm9aNQ 
提取码：129j 
复制这段内容后打开百度网盘手机App，操作更方便哦

一、实验内容

手势翻译
在现代社会，各种语言翻译的应用数不胜数，但是仍然存在一些问题，比如聋哑人和非聋哑人之间的交流问题和聋哑人使用智能助手问题，现在的智能助手大多只可以进行语音识别，不具备手势翻译功能。基于以上考虑，可以考虑开发一类应用，它能够将手势翻译成可读可翻译的语言，这样便可以利用已有的软件为大众服务。

实验要求：

1 使用CNN，RNN等常用的深度学习算法和TensorFlow等常见的深度学习框架进行模型搭建
2 打开摄像头，实时预测手势语言


二、数据集简介

由于网上此类数据集的局限性，下载了共1500多张手势图片，图片的手势分为5类，分别为Nothing、Peace、Ok、Punch和Stop，每张图片的命名就是其标签，每类标签的含义如下：

标签	含义

Peace	很好、很安全

Ok	可以

Punch	有危险、有风险

Stop	停止

Nothing	非以上的标签


具体实现过程见代码过程
