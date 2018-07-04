import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

learningrate=0.002
epochs=100

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
train_data.info()
# 训练集处理
features = train_data[['Age','Survived','Fare','Parch','SibSp','Pclass']]
print(features)
features_notnull = features.loc[(train_data.Age.notnull())]
features_isnull = features.loc[(train_data.Age.isnull())]
X = features_notnull.values[:,1:]
Y = features_notnull.values[:,0]
rfr = RandomForestRegressor(n_estimators=1000,n_jobs=-1)
rfr.fit(X,Y)
predictAges = rfr.predict(features_isnull.values[:,1:])
train_data.loc[(train_data.Age.isnull()),'Age'] = predictAges

train_data.loc[train_data['Sex']=='male','Sex'] = 0
train_data.loc[train_data['Sex']=='female','Sex'] = 1

train_data['Embarked'] = train_data['Embarked'].fillna('S')
train_data.loc[train_data['Embarked'] == 'S','Embarked'] = 0
train_data.loc[train_data['Embarked'] == 'C','Embarked'] = 1
train_data.loc[train_data['Embarked'] == 'Q','Embarked'] = 2
train_data['Cabin'] = train_data['Cabin'].fillna('000')
train_data.loc[train_data['Cabin']=='000','Cabin']=0
train_data.loc[train_data['Cabin']!='000','Cabin']=1
# train_data.drop(['Cabin'],axis=1,inplace=True)
train_data['Deceased'] = train_data['Survived'].apply(lambda s: 1 - s)
train_data.info()

train_data['Family']=train_data['Parch']+train_data['SibSp']
dataset_X = train_data[['Sex','Age','Pclass','Family','Fare','Cabin']]
dataset_Y = train_data[['Deceased','Survived']]


X_train,X_val,Y_train,Y_val = train_test_split(dataset_X.as_matrix(),
                                                 dataset_Y.as_matrix(),
                                                test_size = 0.2,
                                                random_state = 42)

x = tf.placeholder(tf.float32,shape = [None,6],name = 'input')
y = tf.placeholder(tf.float32,shape = [None,2],name = 'label')
weights1 = tf.Variable(tf.random_normal([6,6]),name = 'weights1')
bias1 = tf.Variable(tf.zeros([6]),name = 'bias1')
a = tf.nn.relu(tf.matmul(x,weights1) + bias1)
weights2 = tf.Variable(tf.random_normal([6,2]),name = 'weights2')
bias2 = tf.Variable(tf.zeros([2]),name = 'bias2')
z = tf.matmul(a,weights2) + bias2
y_pred = tf.nn.softmax(z)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=z))
correct_pred = tf.equal(tf.argmax(y,1),tf.argmax(y_pred,1))
acc_op = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
train_op = tf.train.AdamOptimizer(learningrate).minimize(cost)

# 存档入口
saver = tf.train.Saver()

# 在Saver声明之后定义的变量将不会被存储
# non_storable_variable = tf.Variable(777)

ckpt_dir = './ckpt_dir'
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    ckpt = tf.train.latest_checkpoint(ckpt_dir)
    if ckpt:
        print('Restoring from checkpoint: %s' % ckpt)
        saver.restore(sess, ckpt)

    for epoch in range(epochs):
        total_loss = 0.
        for i in range(len(X_train)):
            feed_dict = {x: [X_train[i]],y:[Y_train[i]]}
            _,loss = sess.run([train_op,cost],feed_dict=feed_dict)
            total_loss +=loss
        print('Epoch: %4d, total loss = %.12f' % (epoch,total_loss))
        if epoch % 10 == 0:
            accuracy = sess.run(acc_op,feed_dict={x:X_val,y:Y_val})
            print("Accuracy on validation set: %.9f" % accuracy)
            saver.save(sess, ckpt_dir + '/logistic.ckpt')
    print('training complete!')

    accuracy = sess.run(acc_op,feed_dict={x:X_val,y:Y_val})
    print("Accuracy on validation set: %.9f" % accuracy)
    pred = sess.run(y_pred,feed_dict={x:X_val})
    correct = np.equal(np.argmax(pred,1),np.argmax(Y_val,1))
    numpy_accuracy = np.mean(correct.astype(np.float32))
    print("Accuracy on validation set (numpy): %.9f" % numpy_accuracy)

    saver.save(sess, ckpt_dir + '/logistic.ckpt')

    '''
    测试数据的清洗和训练数据一样，两者可以共同完成
    '''

    #数据清洗, 数据预处理
    test_data.loc[test_data['Sex']=='male','Sex'] = 0
    test_data.loc[test_data['Sex']=='female','Sex'] = 1

    features = test_data[['Age','Sex','Parch','SibSp','Pclass']]
    features_notnull = features.loc[(test_data.Age.notnull())]
    features_isnull = features.loc[(test_data.Age.isnull())]
    X = features_notnull.values[:,1:]
    Y = features_notnull.values[:,0]
    rfr = RandomForestRegressor(n_estimators=1000,n_jobs=-1)
    rfr.fit(X,Y)
    predictAges = rfr.predict(features_isnull.values[:,1:])
    test_data.loc[(test_data.Age.isnull()),'Age'] = predictAges

    test_data['Embarked'] = test_data['Embarked'].fillna('S')
    test_data.loc[test_data['Embarked'] == 'S','Embarked'] = 0
    test_data.loc[test_data['Embarked'] == 'C','Embarked'] = 1
    test_data.loc[test_data['Embarked'] == 'Q','Embarked'] = 2
    test_data['Cabin'] = test_data['Cabin'].fillna('000')
    test_data.loc[test_data['Cabin'] == '000', 'Cabin'] = 0
    test_data.loc[test_data['Cabin'] != '000', 'Cabin'] = 1
    # test_data.drop(['Cabin'],axis=1,inplace=True)

    #特征选择
    test_data['Family']=test_data['Parch']+test_data['SibSp']
    X_test = test_data[['Sex', 'Age', 'Pclass', 'Family', 'Fare','Cabin']]

    #评估模型
    predictions = np.argmax(sess.run(y_pred, feed_dict={x: X_test}), 1)

    #保存结果
    submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": predictions
    })
    submission.to_csv("./Result/2-layer-nn3.csv", index=False)