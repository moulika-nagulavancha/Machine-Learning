# Breast Cancer prediction Linear Regression with existing dataset in sklearn package
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

# built in datasets available in sklearn
breast_cancer_data = datasets.load_breast_cancer()

# slice the data to use only one feature
# ['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module']
# print(diabetes.keys())
# print(diabetes.DESCR)

# extracting single feature to plot on graph
breast_cancer_data_X = breast_cancer_data.data
# print(diabetes_X)

# slicing train -> last 30 test -> first 30
# X-axis label
breast_cancer_data_X_train = breast_cancer_data_X[:-30]
breast_cancer_data_X_test = breast_cancer_data_X[-30:]

# Y-axis target label
breast_cancer_data_Y_train = breast_cancer_data.target[:-30]
breast_cancer_data_Y_test = breast_cancer_data.target[-30:]

# load the Linear regression algo
model = linear_model.LinearRegression();

# feed the model with the train data
model.fit(breast_cancer_data_X_train, breast_cancer_data_Y_train)

# predict the outcome
breast_cancer_data_Y_predict = model.predict(breast_cancer_data_X_test)

# mean squared error metric -> error average value
print("Mean Squared Error is: ", mean_squared_error(breast_cancer_data_Y_test, breast_cancer_data_Y_predict))

# Weights, intercept from model
print("Weights: ", model.coef_)
print("Intercept: ", model.intercept_)

#----------Solution-------------
# Mean Squared Error is:  0.06735853532574244
# Weights:  [ 2.26582106e-01 -6.69136393e-03 -2.44566944e-02 -3.60833964e-04
#  -9.56432298e-01  4.74665726e+00 -1.85367334e+00 -1.62060925e+00
#  -1.59221736e-01 -7.52604646e-01 -4.24374018e-01  1.50158732e-02
#   3.03687898e-02  5.87117501e-04 -1.58149570e+01  6.84609636e-01
#   4.26377472e+00 -1.37541629e+01 -1.77576118e+00  3.01777241e+00
#  -1.91089122e-01 -7.74274469e-03 -1.63774056e-04  1.11430270e-03
#  -1.80147113e-01 -2.03804749e-01 -3.97836101e-01 -2.04987102e-01
#  -5.24224653e-01 -3.94610046e+00]
# Intercept:  3.1760917762173992