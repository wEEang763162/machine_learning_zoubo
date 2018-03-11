# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


if __name__ == '__main__':
    pd.set_option('display.width', 300)

    data = pd.read_csv('..\\tel.csv', skipinitialspace=True, thousands=',')
    print('原始数据：\n', data.head(10))

    le = LabelEncoder()
    for col in data.columns:
        data[col] = le.fit_transform(data[col])

    # 年龄分组
    bins = [-1, 6, 12, 18, 24, 35, 50, 70]
    data['age'] = pd.cut(data['age'], bins=bins, labels=np.arange(len(bins)-1))

    # 取对数
    columns_log = ['income', 'tollten', 'longmon', 'tollmon', 'equipmon', 'cardmon',
                   'wiremon', 'longten', 'tollten', 'equipten', 'cardten', 'wireten', ]
    mms = MinMaxScaler()
    for col in columns_log:
        data[col] = np.log(data[col] - data[col].min() + 1)
        # data[col] = pd.cut(data[col], bins=10, labels=np.arange(10))    # 可不做
        data[col] = mms.fit_transform(data[col].values.reshape(-1, 1))

    # one-hot编码
    # marital/retire/gender/tollfree/equip/callcard/wireless/multline/voice/pager/internet/callwait/forward/confer/ebill
    columns_one_hot = ['region', 'age', 'address', 'ed', 'reside', 'custcat']
    for col in columns_one_hot:
        data = data.join(pd.get_dummies(data[col], prefix=col))

    data.drop(columns_one_hot, axis=1, inplace=True)

    columns = list(data.columns)
    columns.remove('churn')
    x = data[columns]
    y = data['churn']
    print('分组与One-Hot编码后：\n', x.head(10))

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=0)

    clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=12, min_samples_split=5,
                                 oob_score=True, class_weight={0: 1, 1: 1/y_train.mean()})
    clf.fit(x_train, y_train)

    # 特征选择
    important_features = pd.DataFrame(data={'features': x.columns, 'importance': clf.feature_importances_})
    important_features.sort_values(by='importance', axis=0, ascending=False, inplace=True)
    important_features['cum_importance'] = important_features['importance'].cumsum()
    print('特征重要度：\n', important_features)
    selected_features = important_features.loc[important_features['cum_importance'] < 0.95, 'features']

    # 重新组织数据
    x_train = x_train[selected_features]
    x_test = x_test[selected_features]

    # 模型训练
    clf.fit(x_train, y_train)
    print('OOB Score: ', clf.oob_score_)
    y_train_pred = clf.predict(x_train)

    print('训练集准确率：', accuracy_score(y_train.values, y_train_pred))
    print('训练集查准率：', precision_score(y_train, y_train_pred))
    print('训练集查全率：', recall_score(y_train, y_train_pred))
    print('训练集f1 Score：', f1_score(y_train, y_train_pred))

    y_test_pred = clf.predict(x_test)
    print('训练集准确率：', accuracy_score(y_test, y_test_pred))
    print('训练集查准率：', precision_score(y_test, y_test_pred))
    print('训练集查全率：', recall_score(y_test, y_test_pred))
    print('训练集f1 Score：', f1_score(y_test, y_test_pred))
