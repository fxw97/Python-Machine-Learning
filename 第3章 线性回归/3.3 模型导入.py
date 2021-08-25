from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

boston = load_boston()
x_train, x_test, y_train, y_test = train_test_split(boston.data,boston.target,test_size=0.2,random_state=22)

estimator = joblib.load('3.3 linear model.pkl')
print('这个模型的偏置是：\n',estimator.intercept_)
print('这个模型的系数是：\n',estimator.coef_)

y_pred = estimator.predict(x_test)
print('预测值是：\n',y_pred)

ret = mean_squared_error(y_test,y_pred)
print('均方误差:\n',ret)