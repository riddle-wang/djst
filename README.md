#### Dynamic Joint Sentiment-Topic (DJST) Model

Written by: Ruidong Wang, wantrd@yeah.net <br>
Part of code is from :

- http://gibbslda.sourceforge.net/. 
- https://github.com/linron84/JST

This file is part of DJST implementation.

#### Outline

- Introduction
- Config Parameter
- Run the program
- Data format

#### Introduction

This is a C++ implementation of the Dynamic joint sentiment-topic(DJST) model for text modeling. The algorithm model comes from the paper[3].

DJST is the variant of LDA(Latent Dirichlet Allocation), which supplement historical data and sentiment to the original topic model.

#### Config Parameter

training.properties

```
nsentiLabs=3
ntopics=10
niters=300 // Max iteration times
savestep=100
updateParaStep=40
twords=10
alpha=0.01
beta=0.01
gamma=0.01
max_epochs=10
S=3 // Number of preceding epoch
input_dir=/Your_path/dJST/input/training/
output_dir=/Your_path/dJST/output/training/
sentiment_lexicon_dir=/Your_path/dJST/input/sentiment-lexicon/chinese/
train_file_list=trainlist.txt
positive_lexicon=positive.txt
negative_lexicon=negative.txt

```


inference.properties

```
input_dir=/Your_path/dJST/input/inference/
output_dir=/Your_path/dJST/output/inference/
sentiment_lexicon_dir=/Your_path/dJST/input/sentiment-lexicon/chinese/
model_dir=/Your_path/dJST/output/training/
model=t10-iter200 // Model name without suffix
model_wordmap=wordmap.txt
datasetFile=1.txt
positive_lexicon=positive.txt
negative_lexicon=negative.txt
niters=1000
savestep=200
twords=20
beta=0.01
alpha=0.01
gamma=0.01
```

#### Run the program

Estimation

```
$ dJST -est config /Your_Path/training.properties
```

Inference

```
$ dJST -inf config /Your_Path/inference.properties
```

#### Data format

Sentiment Lexicon

```
Consist of two files:
    - positive.txt
    - negative.txt
Note: each word only have three kind of sentiment labels: Neural, Positive, Negative
```

trainlist.txt

```
Contain train file name list, the files are sorted by time;

```
traindata

```
- A file represents an epoch data;
- An epoch data include a lot of document;
- A document takes up one row, and the word seperated by one space;
- The first word in the document is the document_id
```

----------------------- 中文版说明 -----------------------------

#### 提纲

- 简介
- 参数设置
- 执行程序
- 数据格式

#### 简介

本文是C++实现的DJST（Dynamic joint sentiment-topic）用于文本建模的模型，算法介绍请参考论文[3]。

DJST 是LDA文本建模的变形，其考虑了历史数据和情感因素的影响；

#### 参数设置

training.properties

```
nsentiLabs=3
ntopics=10
niters=300 // 最大迭代次数
savestep=100
updateParaStep=40
twords=10
alpha=0.01
beta=0.01
gamma=0.01
max_epochs=10
S=3 // 考虑多少个单位的历史数据（历史数据以epoch为跨度单位）
input_dir=/Your_path/dJST/input/training/
output_dir=/Your_path/dJST/output/training/
sentiment_lexicon_dir=/Your_path/dJST/input/sentiment-lexicon/chinese/
train_file_list=trainlist.txt
positive_lexicon=positive.txt
negative_lexicon=negative.txt

```


inference.properties

```
input_dir=/Your_path/dJST/input/inference/
output_dir=/Your_path/dJST/output/inference/
sentiment_lexicon_dir=/Your_path/dJST/input/sentiment-lexicon/chinese/
model_dir=/Your_path/dJST/output/training/
model=t10-iter200 // 不带后缀的模型名字（从训练结果中读取）
model_wordmap=wordmap.txt
datasetFile=1.txt
positive_lexicon=positive.txt
negative_lexicon=negative.txt
niters=1000
savestep=200
twords=20
beta=0.01
alpha=0.01
gamma=0.01
```

#### 执行程序

Estimation

```
$ dJST -est config /Your_Path/training.properties
```

Inference

```
$ dJST -inf config /Your_Path/inference.properties
```

#### 数据格式

Sentiment Lexicon

```
包含两类文件:
    - positive.txt
    - negative.txt
Note: 每个词只有三种情感可选: Neural（中性）, Positive（积极）, Negative（消极）
```

trainlist.txt

```
包含训练数据文件名称，文件名称按照时间排序，每个文件名占据1行；

```
traindata

```
- 1个文件表示1个epoch（它是历史数据的最小单位）；
- 每个epoch包含多个document；
- 每个document占据一行，document内的语句，由单词组成，词之间由1个空格隔开；
- document中第一个词，是document id；
```


Reference:

[1] https://github.com/linron84/JST <br>
[2] http://gibbslda.sourceforge.net/. <br>
[3] He Y, Lin C, Gao W, et al. Dynamic joint sentiment-topic model[J]. Acm Transactions on Intelligent Systems & Technology, 2014, 5(1):1-21. <br>
