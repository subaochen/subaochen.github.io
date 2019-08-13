---
title: 使用tensorflow lite遇到的一些坑
type: post
categories:
- android
layout: post
date: 2019-08-12
tags: [android,tensorflow lite]
status: publish
published: true
comments: true
---

尽管[这里](https://tensorflow.google.cn/lite)有tensorflow lite的详细
介绍，在实战中还是不免要踩一些坑。下面是我踩过的坑，记录下来，免得下次踩同样的坑）。

测试环境：

* 自己训练的模型。使用预训练的模型应该会更简单一些，不需要模型转换这个步骤。
* Tensorflow 2.0.0-beta1
* Android Studio 3.4.2
* 简单的回归预测案例

# 模型转换

试验了独立程序和训练后即时转换两种方式，最后觉得在训练模型后马上进行模型转换更方便，原因有两个：

1. 在训练程序中，可以直接使用模型对象model进行模型的转换。
2. 通常输入数据要进行标准化处理，而每次训练时所采用的数据集是不同的，导致标准化数据的mean和std会随之变化，因此需要将标准化数据的mean和std保存下来，以便传递给android app对输入数据进行同样的数据标准化处理。

上面的两个数据：转换后的模型和数据标准化基础数据都需要复制到Android app的assets目录下，因此在训练的程序中统一写一下更加方便，下面是我这边相应的代码：

```python
# 模型训练完毕后转化模型
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("mymodel.tflite", "wb").write(tflite_model)
```

```python
# 保存数据标准化相关数据（主要是mean和std）
train_stats = train_dataset.describe()
train_stats = train_stats.transpose()
# 创建csv文件，以便移动端使用相同的统计数据标准化数据
train_stats.to_csv('train_stats.csv', index=False)
```

# Android App的配置

Android Studio这一端的配置涉及到以下几个方面：

## 模型和数据标准化基准数据的存放位置

转化后的模型和数据标准化基准数据一般要保存到`app/src/main/assets`目录下，方便在程序中引用。

## 依赖的引入

在app/build.gradle文件中增加如下的依赖：

```
dependencies {
    ......
    implementation 'org.tensorflow:tensorflow-lite:0.0.0-nightly'
    implementation 'org.tensorflow:tensorflow-lite-gpu:0.0.0-nightly'
    implementation 'com.opencsv:opencsv:4.5'
    ......
}
```

同样的，在app/build.gradle文件中增加如下的片段，原因已经在注释中说明了：

```
android {
    compileSdkVersion 28

    // aapt默认压缩assets下面的文件，直接openFd打不开
    aaptOptions {
        noCompress "tflite"  //表示不让aapt压缩的文件后缀
    }
......
}
```

## Interpreter的创建和使用

有了以上的准备工作，就可以在适当的Action中使用Interpreter来运行模型了：

```java
private static final String MODEL = "mymodel.tflite";
......
try (Interpreter interpreter = new Interpreter(loadModelFile(MODEL))) {
        ......
        interpreter.run(normed_input, output);
}

    /**
     * Memory-map the model file in Assets.
     *
     * @see https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/java/demo/app/src/main/java/com/example/android/tflitecamerademo/ImageClassifier.java
     */
private MappedByteBuffer loadModelFile(String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = getAssets().openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
}
```

# 输入/输出数据的维度和标准化处理

Interpreter的参数很简单，一个是input，一个output，但是要注意这两个参数的维度和类型。input只能使用Interpreter能够识别的基本数据类型（int,float,long,byte）数组，output的类型和维度因程序而异。input的类型和维度在训练程序中查看更加方便，因此一般要在训练程序中搞清楚input和output的类型和维度再来写android端的程序，此时你会发现，python在AI程序方面的优势太大了。

我这边的情况input和output都是float数组：

```java
    private float[][] input;
    private float[][] output;
```

在上面说过，喂入模型的数据一般要经过标准化处理，那么在android端的输入数据也要进行同样的标准化处理，这在android端是个有点麻烦的事情。我采取的方法是读取模型训练时的标准化基准数据（主要是mean和std）csv文件，然后对输入数据使用mean和std进行标准化处理：

```java
            private static final String MEAN_STD = "train_stats.csv";
            
            private float[][] norm(float[][] input) {
                float[][] normed_input = new float[input.length][43];
                float[][] mean_std = new float[43][2];
                CSVReader reader = null;
                try {
                    reader = new CSVReader(new BufferedReader(new InputStreamReader(getAssets().open(MEAN_STD))));
                    List<String[]> myEntries = reader.readAll();
                    int index = 0;
                    for(String[] entry:myEntries){
                        float mean = Float.parseFloat(entry[1]);
                        float std = Float.parseFloat(entry[2]);
                        mean_std[index][0] = mean;
                        mean_std[index][1] = std;
                        index++;
                    }

                    for(int i = 0; i < input.length; i++){
                        for(int j = 0; j < 43; j++){
                            normed_input[i][j] = (input[i][j] - mean_std[j][0])/mean_std[j][1];
                        }
                    }


                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                } catch (IOException e) {
                    e.printStackTrace();
                }

                return normed_input;
            }
```

