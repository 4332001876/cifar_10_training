0 batch_size是不是越大越好？
0 Answer: 是/不是
当batch_size为16，32，64，128时，测试集上的准确率分别是A，B, C, D, E
可能是原因是：
1 在训练集那里的transform试一下RandomHorizontalFlip，效果会更好吗？
2 换一个optimizer, 使效果更好一些
3 保持epoch数不变，加一个scheduler，是否能让效果更好一些
4 根据Net() 生成 Net1(), 加入三个batch_normalization层，显示测试结果
5 根据Net() 生成Net2(), 使用Kaiming初始化卷积与全连接层，显示测试结果
6 根据Net()生成Net3(),将Net()中的通道数加到原来的2倍，显示测试结果
7 在不改变Net()的基础结构（卷积层数、全连接层数不变）和训练epoch数的前提下，你能得到最好的结果是多少？
8 使用ResNet18(),显示测试结果