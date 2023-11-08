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

Answer: 
我们对数据集做了RandomHorizontalFlip的Augmentation，改动后在测试集上的准确率为56%，与原来相比没有明显变化
其Training Loss在最后一个epoch平均为1.226，相较原来的版本中的1.209高了一点，这是在预期之中的，因为增加Augmentation之后过拟合的程度降低了，Training Loss会提高
实际上，由于这个Augmentation是比较弱的，所以效果并不明显



## 2 
换一个optimizer, 使效果更好一些

Answer: 我们可以使用Adam来大幅优化训练效率
```python
self.optimizer = torch.optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)
```
用其训练后，收敛速度大幅提高，且最终精度也大幅提高：

对比它与原来的SGD在每个epoch的表现如下：
| epoch |  SGD  | Adam  |
| :---: | :---: | :---: |
|   1   |  16%  |  48%  |
|   2   |  29%  |  54%  |
|   3   |  36%  |  57%  |
|   4   |  40%  |  60%  |
|   5   |  43%  |  62%  |
|   6   |  46%  |  62%  |
|   7   |  48%  |  63%  |
|   8   |  50%  |  64%  |
|   9   |  51%  |  63%  |
|  10   |  53%  |  65%  |
|  11   |  55%  |  64%  |
|  12   |  56%  |  65%  |


## 3 
保持epoch数不变，加一个scheduler，是否能让效果更好一些

Answer: 
我们使用最简单的StepLR来进行学习率的调整，我们设置的参数令其每过5个epoch学习率减半，初始学习率设为0.002，分别对SGD与Adam测试如下：
|    Group     | SGD-Accuracy | SGD-Loss | Adam-Accuracy | Adam-Loss |
| :----------: | :----------: | :------: | :-----------: | :-------: |
| No-Scheduler |     56%      |  1.211   |      65%      |   0.779   |
|   Step-LR    |     60%      |  1.068   |      65%      |   0.668   |

可以看到，使用StepLR后，SGD的效果有了一定的提升，而Adam的效果没有明显变化，但两个优化器训出来的Loss都有了明显的下降，这说明了学习率的调整是有效的，而Adam的效果没有明显变化可能是因为过拟合了。


## 4 
根据Net() 生成 Net1(), 加入三个batch_normalization层，显示测试结果

Answer: 
在两个卷积层和第一个线性层后加入BN层后，在测试集上的准确率为65%，较原来提升了9%，可见BatchNorm对效果的提升极为显著

具体代码如下：
```python
def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.bn1(x)
    x = self.pool(F.relu(self.conv2(x)))
    x = self.bn2(x)
    x = torch.flatten(x, 1) # flatten all dimensions except batch
    x = F.relu(self.fc1(x))
    x = self.bn3(x)
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x
```

## 5 
根据Net() 生成Net2(), 使用Kaiming初始化卷积与全连接层，显示测试结果

Answer: 
加入Kaiming初始化后，在测试集上的准确率为57%，较原来提升了1%。

实际上，`nn.Linear`及`nn.Conv2d`的父类`nn._ConvNd`的初始化默认就使用了Kaiming初始化，它们初始化时都调用了以下函数：
```python
def reset_parameters(self) -> None:
    # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
    # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
    # https://github.com/pytorch/pytorch/issues/57109
    init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.bias, -bound, bound)
```

此处的提升可能是因为Kaiming初始化的参数不一样。

## 6 
根据Net()生成Net3(),将Net()中的通道数加到原来的2倍，显示测试结果

Answer: 
通道数翻倍后，在测试集上的准确率为60%，较原来提升了4%，可见提升通道数有一定效果。

## 7 
在不改变Net()的基础结构（卷积层数、全连接层数不变）和训练epoch数的前提下，你能得到最好的结果是多少？

Answer: 
我们进行了如下改进：
- 使用Adam优化器，增加了0.0001的WEIGHT_DECAY，使用`CosineAnnealingLR`余弦退火学习率调整器，初始学习率设为0.002
- 同上述BN网络加入三个BatchNorm层，并加入了三个p=0.1的dropout层，这三个层正好加在BatchNorm层之后
- 前两层卷积层通道数分别改为256与256
- 采用了大量数据增强，具体如下（其中AutoAugment()来自论文'AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty - https://arxiv.org/abs/1912.02781）：
```python
def get_transform(self):
    res = []
    res.append(transforms.RandomHorizontalFlip(p=0.5))
    res.extend([transforms.Pad(2, padding_mode='constant'),
                    transforms.RandomCrop([32,32])])
    res.append(transforms.RandomApply([AutoAugment()], p=0.6))
    res.append(transforms.ToTensor())
    res += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(res)
```

最终在测试集上的准确率为83%，较原来提升了27%。

## 8 
使用ResNet18(),显示测试结果

Answer: 
使用了大名鼎鼎的resnet-18之后

我们顺便测试了pretrained的resnet-18的精度，我们将最后一层替换为10维，进行finetune后，
发现在测试集上的准确率为：
