# statistical-learning-method

李航《统计学习方法》算法实现

用python手动实现和sklearn实现《统计学习方法》中所提到的算法 


第一章：最小二乘法

第二章：[感知机（Perceptron）](https://github.com/zhengjingwei/statistical-learning-method/tree/master/Perceptron)

第三章：[K近邻（k-NN）](https://github.com/zhengjingwei/statistical-learning-method/tree/master/KNN)

第四章：[朴素贝叶斯（Naive Bayes）](https://github.com/zhengjingwei/statistical-learning-method/tree/master/NaiveBayes)

第五章：[决策树（Decision Tree）](https://github.com/zhengjingwei/statistical-learning-method/tree/master/DecisionTree)

第六章：[逻辑斯蒂回归（Logistic Regression）](https://github.com/zhengjingwei/statistical-learning-method/tree/master/LogisticRegression)

​		[最大熵模型（Maximum Entropy Model）](https://github.com/zhengjingwei/statistical-learning-method/tree/master/MaxEntropy)

第七章：[支持向量机（SVM）](https://github.com/zhengjingwei/statistical-learning-method/tree/master/SVM)

​     

**实验数据**：MNIST数据集,这里用kaggle中处理好的数据 

官方下载地址：http://yann.lecun.com/exdb/mnist/ 

kaggle中处理好的数据：https://www.kaggle.com/c/digit-recognizer/data

数据集说明：

我们将train.csv作为完整数据集， 随机选取33%数据作为测试集，剩余为训练集。数据集共有0到9这10个类别的数据。每个样本由28x28的像素构成，每个像素是一个0-255的灰度值。

- 数据集集大小42000
- 特征维度784
- 数据集第一列为label，第二列到最后一列为特征


对于二分类问题，将MINST数据集train.csv的label列进行了一些微调，label等于0的继续等于0，label大于0改为1。这样就将十分类的数据改为二分类的数据 data/train_binary.csv