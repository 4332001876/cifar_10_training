# Deep Learning Lab Report - CIFAR-10

## 0 
batch_size是不是越大越好？

Answer: 不是
使用不同的batch_size，在测试集上的准确率分别为：
| batch_size | accuracy |
| :--------: | :------: |
|     16     |   65%    |
|     32     |   63%    |
|     64     |   56%    |
|    128     |   47%    |

这说明了batch_size并不是越大越好。

可能的原因是：
- 大的batch会导致模型更容易收敛到局部最优解，而不是全局最优解；小的batch实际起了退火的作用
- 大的batch会导致模型更新次数减少，收敛速度变慢
  
另外，大的batch也可能导致显存容量不足，无法训练



## 1 
在训练集那里的transform试一下RandomHorizontalFlip，效果会更好吗？

Answer: 我们对数据集做了RandomHorizontalFlip的Augmentation，改动后在测试集上的准确率为：

## 2 
换一个optimizer, 使效果更好一些

Answer: 我们可以使用Adam来大幅优化训练效率
```python

```

## 3 
保持epoch数不变，加一个scheduler，是否能让效果更好一些

Answer: 
我们使用最简单的StepLR来进行学习率的调整，效果如下：

## 4 
根据Net() 生成 Net1(), 加入三个batch_normalization层，显示测试结果

Answer: 

## 5 
根据Net() 生成Net2(), 使用Kaiming初始化卷积与全连接层，显示测试结果

Answer: 

## 6 
根据Net()生成Net3(),将Net()中的通道数加到原来的2倍，显示测试结果

Answer: 

## 7 
在不改变Net()的基础结构（卷积层数、全连接层数不变）和训练epoch数的前提下，你能得到最好的结果是多少？

Answer: 

## 8 
使用ResNet18(),显示测试结果

Answer: 
使用了大名鼎鼎的resnet-18之后

我们顺便测试了pretrained的resnet-18的精度，我们将最后一层替换为10维，进行finetune后，
发现在测试集上的准确率为：
