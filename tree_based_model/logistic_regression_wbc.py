# Logistic Regression on Wisconsin Data
import pandas as pd
from sklearn.model_selection import train_test_split
# Import LogisticRegression from sklearn.linear_model
from sklearn.linear_model import  LogisticRegression
from sklearn.metrics import accuracy_score

SEED = 1

# Read the dataset
df = pd.read_csv('./tree_based_model/wbc.csv')

# drop unwanted column from dataset
df = df.drop('id', axis = 1)

# Using only 2 features from dataset
y = df['diagnosis']
X = df[['radius_mean', 'concave points_mean']]

# convert M to 1 and B to 0
y = y.apply(lambda x : 1 if x == 'M' else 0)

# split data in train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = SEED)

# Instatiate logreg
logreg = LogisticRegression(random_state=1)

# Fit logreg to the training set
logreg.fit(X_train, y_train)

# predict and calculate score
y_pred = logreg.predict(X_test)
log_score = accuracy_score(y_test, y_pred)

print('Logistic Regression Accuracy : {}'.format(log_score))
