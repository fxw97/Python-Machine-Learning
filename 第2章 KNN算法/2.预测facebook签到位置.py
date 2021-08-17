import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# 缩小原始数据范围
# facebook = pd.read_csv(r'F:\BaiduNetdiskDownload\资料-边学边练超系统掌握人工智能机器学习算法\day05-资料\2.code\data\FBlocation\train.csv')
# facebook = facebook.query('x>2.0 & x<2.5 & y>2.0 & y<2.5')
# facebook.to_csv('2.facebook_data.csv')

data = pd.read_csv('2.facebook_data.csv')

# 1 选择时间特征
time = pd.to_datetime(data['time'],unit='s')
time = pd.DatetimeIndex(time)
data['day'] = time.day
data['hour'] = time.hour
data['weekday'] = time.weekday

# 2.3 去掉签到较少的地方
place_count = data.groupby('place_id').count()
place_count = place_count[place_count['row_id']>3]
data = data[data['place_id'].isin(place_count.index)]

# 2.4 确定特征值和目标值
x = data[['x','y','accuracy','day','hour','weekday']]
y = data['place_id']

# 2.5 分隔数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22)

# 3 特征工程——特征值预处理(标准化)
# 3.1 实例化一个转换器
transfer = StandardScaler()
# 3.2 调用fit_transform
x_train = transfer.fit_transform(x_train)
x_test = transfer.fit_transform(x_test)

# 4 机器学习--KNN+cv
# 4.1 实例化一个估计器
estimator = KNeighborsClassifier()
# 4.2 调用gridsearchCV
param_grid = {'n_neighbors':[1,3,5,7,9]}
estimator = GridSearchCV(estimator,param_grid=param_grid, cv=5, n_jobs=-1) # n_job表示用几个CPU来跑程序，可以传入个数，-1表示用所有的cpu; cv指定几折交叉验证
# 4.3 模型训练
estimator.fit(x_train, y_train)

# 5 模型评估
# 5.1 基本评估方式
score = estimator.score(x_test, y_test)
print("最后预测的准确率为:\n", score)

y_predict = estimator.predict(x_test)
print("最后的预测值为:\n", y_predict)
print("预测值和真实值的对比情况:\n", y_predict == y_test)

# 5.2 使用交叉验证后的评估方式
print("在交叉验证中验证的最好结果:\n", estimator.best_score_)
print("最好的参数模型:\n", estimator.best_estimator_)
print("每次交叉验证后的验证集准确率结果和训练集准确率结果:\n",estimator.cv_results_)