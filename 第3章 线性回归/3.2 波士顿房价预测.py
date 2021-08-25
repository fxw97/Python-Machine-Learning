'''
1. 获取数据
2. 数据基本处理
2.1 分割数据
3. 特征工程-标准化
4. 机器学习-线性回归
5. 模型评估
'''
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn.metrics import mean_squared_error


def linear_model1():
    boston = load_boston()
    x_train, x_test, y_train, y_test = train_test_split(boston.data,boston.target,test_size=0.2)

    transfer = StandardScaler()
    transfer.fit_transform(x_train)
    transfer.fit_transform(x_test)

    estimator = LinearRegression()
    estimator.fit(x_train, y_train)

    print('这个模型的偏置是：\n',estimator.intercept_)
    print('这个模型的系数是：\n',estimator.coef_)

    y_pred = estimator.predict(x_test)
    print('预测值是：\n',y_pred)

    ret = mean_squared_error(y_test,y_pred)
    print('均方误差:\n',ret)


def linear_model2():
    boston = load_boston()
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2)

    transfer = StandardScaler()
    transfer.fit_transform(x_train)
    transfer.fit_transform(x_test)

    estimator = SGDRegressor(max_iter=1000) # 修改模型为梯度下降法求回归
    estimator.fit(x_train, y_train)

    print('这个模型的偏置是：\n', estimator.intercept_)
    print('这个模型的系数是：\n', estimator.coef_)

    y_pred = estimator.predict(x_test)
    print('预测值是：\n', y_pred)

    ret = mean_squared_error(y_test, y_pred)
    print('均方误差:\n', ret)

linear_model1()