import pandas as pd
from sklearn.model_selection import train_test_split
# Import DecisionTreeClassifier from sklearn.tree
from sklearn.tree import DecisionTreeClassifier
# Import accuracy_score
from sklearn.metrics import accuracy_score

SEED = 1

# Read the dataset
df = pd.read_csv('./tree_based_model/wbc.csv')

# drop unwanted column from dataset
df = df.drop('id', axis = 1)

# Using only 2 features from dataset
y = df['diagnosis']
X = df[['radius_mean', 'concave points_mean']]

# split data in train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = SEED)

# Instantiate a DecisionTreeClassifier 'dt' with a maximum depth of 6
dt = DecisionTreeClassifier(max_depth = 6, random_state=SEED)

# added criterion explicitly
dt_gini = DecisionTreeClassifier(max_depth=8, random_state=SEED, criterion='gini')
dt_entropy = DecisionTreeClassifier(max_depth=8, random_state=SEED, criterion='entropy')

# Fit dt to the training set
dt.fit(X_train, y_train)
dt_gini.fit(X_train, y_train)
dt_entropy.fit(X_train, y_train)

# Predict test set labels
y_pred = dt.predict(X_test)
y_pred_gini = dt_gini.predict(X_test)
y_pred_entropy = dt_entropy.predict(X_test)

# Compute test set accuracy  
acc = accuracy_score(y_test, y_pred)
print("Test set accuracy: {:.2f}".format(acc))

acc_gini = accuracy_score(y_test, y_pred_gini)
print("Test set accuracy ( gini): {:.2f}".format(acc_gini))

acc_entropy = accuracy_score(y_test, y_pred_entropy)
print("Test set accuracy (entropy): {:.2f}".format(acc_entropy))

# Test set accuracy: 0.89
# Test set accuracy ( gini): 0.89
# Test set accuracy (entropy): 0.89