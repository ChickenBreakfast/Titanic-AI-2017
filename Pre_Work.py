import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series,DataFrame
import seaborn as sns

# 导入并展示训练数据和测试数据

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# 数据概览
train.info()
print('--------------------------------------')
test.info()

# 幸存者分布
plt.subplot2grid((2,3),(0,0))             # 在一张大图里分列几个小图
train.Survived.value_counts().plot(kind='bar')# 柱状图
plt.title('SurvivedStatistics (1 for survived)') # 标题
plt.ylabel('NUMBER OF PEOPLE')

# 乘客性别分布
plt.subplot2grid((2,3),(0,1))
train.Sex.value_counts().plot(kind="bar")
plt.title('Sex')
plt.ylabel('NUMBER OF PEOPLE')

# 乘客座位等级分布
plt.subplot2grid((2,3),(0,2))
train.Pclass.value_counts().plot(kind="bar")
plt.title('Pclass')
plt.ylabel('NUMBER OF PEOPLE')

# 年龄分布
plt.subplot2grid((2,3),(1,0))
plt.scatter(train.Survived, train.Age)
plt.grid(b=True, which='major', axis='y')
plt.title('AgeStatistics(1 for survived)')
plt.ylabel('AGE')

# 各等级的乘客年龄分布
plt.subplot2grid((2,3),(1,1))
train.Age[train.Pclass == 1].plot(kind='kde')
train.Age[train.Pclass == 2].plot(kind='kde')
train.Age[train.Pclass == 3].plot(kind='kde')
plt.title('AgeStatisticsOf3Class')
plt.xlabel('AGE')
plt.ylabel('DENSITY')
plt.legend(('P1', 'P2','P3'),loc='best')

# 各登船口岸上船人数
plt.subplot2grid((2,3),(1,2))
train.Embarked.value_counts().plot(kind='bar')
plt.title('NPStatisticsOfEmbarked')
plt.ylabel('NUMBER OF PEOPLE')
plt.show()

# ------------------------------------------------分割线---------------------------------------------------------

# 座位等级与获救情况
Survived_0 = train.Pclass[train.Survived == 0].value_counts()
Survived_1 = train.Pclass[train.Survived == 1].value_counts()
df = pd.DataFrame({'Survived':Survived_1, 'unSurvived':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title('SurvivedStatisticsByPclass')
plt.xlabel('Pclass')
plt.ylabel('Number Of People')
plt.show()

# 性别与获救情况
Survived_0 = train.Sex[train.Survived == 0].value_counts()
Survived_1 = train.Sex[train.Survived == 1].value_counts()
df = pd.DataFrame({'Survived':Survived_1, 'unSurvived':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title('SurvivedStatisticsBySex')
plt.xlabel('Sex')
plt.ylabel('Number Of People')
plt.show()

# 港口与获救情况
Survived_0 = train.Embarked[train.Survived == 0].value_counts()
Survived_1 = train.Embarked[train.Survived == 1].value_counts()
df = pd.DataFrame({'Survived':Survived_1, 'unSurvived':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title('SurvivedStatisticsByEmbarked')
plt.xlabel('Embarked')
plt.ylabel('Number Of People')
plt.show()

# 船舱有无获救情况
Survived_cabin = train.Survived[pd.notnull(train.Cabin)].value_counts()
Survived_nocabin = train.Survived[pd.isnull(train.Cabin)].value_counts()
df=pd.DataFrame({'Yes':Survived_cabin, 'No':Survived_nocabin}).transpose()
df.columns=['unSurvived','Survived']
temp = df['Survived']
df.drop(labels=['Survived'], axis=1,inplace = True)
df.insert(0, 'Survived', temp)
df.plot(kind='bar', stacked=True)
plt.title('SurvivedStatisticsByCabin')
plt.xlabel('Cabin')
plt.ylabel('Number Of People')
plt.show()

# 年龄与获救情况
facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
plt.show()

# 称呼与获救情况
train['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
nameResult=pd.crosstab(train['Title'], train['Sex'])
print(nameResult)
train[['Title','Survived']].groupby(['Title']).mean().plot.bar()
plt.show()

# 有无兄弟姐妹与获救情况
sibsp_df = train[train['SibSp'] != 0]
no_sibsp_df = train[train['SibSp'] == 0]
plt.figure(figsize=(10,5))
plt.subplot(121)
sibsp_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct = '%1.1f%%')
plt.xlabel('sibsp')
plt.ylabel('')
plt.subplot(122)
no_sibsp_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct = '%1.1f%%')
plt.xlabel('no_sibsp')
plt.ylabel('')
plt.show()

# 有无父母孩子与获救情况
parch_df = train[train['Parch'] != 0]
no_parch_df = train[train['Parch'] == 0]
plt.figure(figsize=(10,5))
plt.subplot(121)
parch_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct = '%1.1f%%')
plt.xlabel('parch')
plt.ylabel('')
plt.subplot(122)
no_parch_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct = '%1.1f%%')
plt.xlabel('no_parch')
plt.ylabel('')
plt.show()

# 家庭与获救情况
fig,ax=plt.subplots(1,2,figsize=(18,8))
train[['Parch','Survived']].groupby(['Parch']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Parch and Survived')
train[['SibSp','Survived']].groupby(['SibSp']).mean().plot.bar(ax=ax[1])
ax[1].set_title('SibSp and Survived')
plt.show()
train['Family_Size'] = train['Parch'] + train['SibSp'] + 1
train[['Family_Size','Survived']].groupby(['Family_Size']).mean().plot.bar()
plt.show()

# 票价与获救情况
fareResult=train['Fare'].describe()
print(fareResult)
fare_not_survived = train['Fare'][train['Survived'] == 0]
fare_survived = train['Fare'][train['Survived'] == 1]
average_fare = pd.DataFrame([fare_not_survived.mean(), fare_survived.mean()])
std_fare = pd.DataFrame([fare_not_survived.std(), fare_survived.std()])
average_fare.plot(yerr=std_fare, kind='bar', legend=False)
plt.show()