# Decision Tree Regression on Auto MPG
import pandas as pd
from sklearn.model_selection import train_test_split
# Import LogisticRegression from sklearn.linear_model
from sklearn.tree import  DecisionTreeRegressor
# Import mean_squared_error from sklearn.metrics as MSE
from sklearn.metrics import mean_squared_error as MSE

SEED = 3

# Read the dataset
df = pd.read_csv('./tree_based_model/auto.csv')
df = pd.get_dummies(df, drop_first=True)

# seperating features and labels
y = df['mpg']
X = df.drop('mpg', axis = 1)

# split data in train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = SEED)

# Create model instance
dt = DecisionTreeRegressor(max_depth=8, min_samples_leaf=0.13, random_state=SEED)

# Fit the model
dt.fit(X_train, y_train)

# Compute y_pred
y_pred = dt.predict(X_test)

# Compute mse_dt
mse_dt = MSE(y_test, y_pred)

# Compute rmse_dt
rmse_dt = mse_dt ** (1/2)

print('RMSE :{} '.format(rmse_dt))
# RMSE :4.366505690565745 
