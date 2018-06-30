from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
import numpy as np
diabetes = datasets.load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]
y = np.mat(y)
print (X)
print (y)
lasso = linear_model.Lasso()
print(cross_val_score(lasso, X, y))  