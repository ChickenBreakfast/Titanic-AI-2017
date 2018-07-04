import numpy as np
# 导入pandas用于数据分析。
import pandas as pd

train_data = pd.read_csv('data/train.csv')
test_data= pd.read_csv('data/test.csv')

test_data['Survived']=0
# combined_train_test=train_data.append(test_data)

#查看字段属性有哪些
# train_data.info()
# test_data.info()

#机器学习有一个不太被初学者重视，并且耗时，但是十分重要的一环，特征的选择，这个需要基于一些背景知识。根据我们对这场事故的了解，sex, age, pclass这些都很有可能是决定幸免与否的关键因素。
# X = combined_train_test[['Pclass', 'Age', 'Sex']]
# y = combined_train_test['Survived']


#Cabin项缺失太多，只能j将Cabin填充处理
train_data['Cabin'] = train_data['Cabin'].fillna('000')
train_data.loc[train_data['Cabin'] == '000', 'Cabin'] = 0
train_data.loc[train_data['Cabin'] != '000', 'Cabin'] = 1

test_data['Cabin'] = test_data['Cabin'].fillna('000')
test_data.loc[test_data['Cabin'] == '000', 'Cabin'] = 0
test_data.loc[test_data['Cabin'] != '000', 'Cabin'] = 1

combined_train_test=train_data.append(test_data)

# 首先我们补充age里的数据，使用平均数或者中位数都是对模型偏离造成最小影响的策略。
combined_train_test['Age'].fillna(combined_train_test['Age'].mean(), inplace=True)

# 数据分割。
train_data=combined_train_test[:891]
test_data=combined_train_test[891:]
X_train=train_data[['Pclass','Age','Sex','Parch','Cabin']]
y_train=train_data['Survived']

X_test=test_data[['Pclass','Age','Sex','Parch','Cabin']]
y_test=test_data['Survived']

# 我们使用scikit-learn.feature_extraction中的特征转换器，详见3.1.1.1特征抽取。
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False)

# 转换特征后，我们发现凡是类别型的特征都单独剥离出来，独成一列特征，数值型的则保持不变。
X_train = vec.fit_transform(X_train.to_dict(orient='record'))

# 同样需要对测试数据的特征进行转换。
X_test = vec.transform(X_test.to_dict(orient='record'))


# 从sklearn.tree中导入决策树分类器。
from sklearn.tree import DecisionTreeClassifier
# 使用默认配置初始化决策树分类器。
dtc = DecisionTreeClassifier()
# 使用分割到的训练数据进行模型学习。
dtc.fit(X_train, y_train)
# 用训练好的决策树模型对测试特征数据进行预测。
y_predict = dtc.predict(X_test)

print(y_predict)
print("------------------------------------------------")
y_test=y_predict
print(y_test)


decision_submission=pd.DataFrame({'PassengerId':test_data['PassengerId'],
	                              'Survived':y_predict[:]})

decision_submission.to_csv('./Result/DecisionTree.csv',index=False,sep=',')