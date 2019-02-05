from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
import seaborn as sb

# loading data
bmd = load_boston()

# dataset description
print("\n")
print("dataset description: \n")
print(bmd['DESCR'])

print("\n")
print("data and target shape")
print(bmd['data'].shape)
print(bmd['target'].shape)

# to pandas
bmd_p = pd.DataFrame(bmd.data, columns=[bmd.feature_names])
# adding target data to pandas dataframe
bmd_p['MEDV'] = bmd.target

# checking if there are any missing values in the data
print("Missing values:\n")
bmd_p.isnull().sum()

# distribution of the target variable MEDV
sb.set(rc={'figure.figsize': (24, 24)})
sb.distplot(bmd_p['MEDV'], bins=30)
plt.show()

# correlation matrix - linear relationships between the variables
correlation_matrix = bmd_p.corr().round(2)
# annot = True to print the values inside the square
sb.heatmap(data=correlation_matrix, annot=True)
plt.show()

# pairplot
sb.pairplot(bmd_p, diag_kind="kde")
plt.show()

# pandas dataset description
print("\n")
print("pandas dataset description: \n")
print(bmd_p.head())
print(bmd_p.describe())

# scaling with StandardScaler
scaler = StandardScaler()
bmd['data'] = scaler.fit_transform(bmd['data'])

# back to pandas
bmd_p = pd.DataFrame(bmd.data, columns=[bmd.feature_names])
# adding target data to pandas dataframe
bmd_p['MEDV'] = bmd.target

# pandas dataset description
print("\n")
print("pandas dataset description AFTER standard scaling: \n")
print(bmd_p.head())
print(bmd_p.describe())

# split into train and batches with train_test_split
bmd_train_data, bmd_test_data, bmd_train_target, bmd_test_target = train_test_split(bmd['data'], bmd['target'],
                                                                                    test_size=.33)
print("\n")
print("Training dataset:")
print("bmd_train_data:", bmd_train_data.shape)
print("bmd_train_target:", bmd_train_target.shape)

print("\n")
print("Testing dataset:")
print("bmd_test_data:", bmd_test_data.shape)
print("bmd_test_target:", bmd_test_target.shape)



# training LinearRegression model
linear_regression = LinearRegression()
linear_regression.fit(bmd_train_data, bmd_train_target)

# using trained model
did = 10
linear_regression_prediction = linear_regression.predict(bmd_test_data[did, :].reshape(1, -1))
print("\n")
print("Model predicted for house {0} value {1}".format(did, linear_regression_prediction))
print("\n")
print("Real value for house {0} is {1}".format(did, bmd_test_target[did]))

# model evaluation
bmd_mean_square_error = mean_squared_error(bmd_test_target, linear_regression.predict(bmd_test_data))
print("\n")
print("Mean square error of a learned model: %.3f " % bmd_mean_square_error)

bmd_r2_score = r2_score(bmd_test_target, linear_regression.predict(bmd_test_data))
print("\n")
print(f"Variance score: %.3f" % bmd_r2_score)
print("\n")
print('Coefficients of a learned model: \n', linear_regression.coef_)

scores = cross_val_score(LinearRegression(), bmd['data'], bmd['target'], cv=4)
print("\n")
print(f"Cross-validation score: {scores}")

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:
predicted = cross_val_predict(linear_regression, bmd['data'], bmd['target'], cv=4)

fig, ax = plt.subplots()
ax.scatter(bmd['target'], predicted, edgecolors=(0, 0, 0))
ax.plot([bmd['target'].min(), bmd['target'].max()], [bmd['target'].min(), bmd['target'].max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
