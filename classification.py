# import the necessary packages
import pandas as pd

# Load the dataset from the csv file using pandas
data = pd.read_csv(‘creditcard.csv’)

#dividing the X and the Y from the dataset
X=data.drop([‘Class’], axis=1)
Y=data[“Class”]
print(X.shape)
print(Y.shape)
#getting just the values for the sake of processing (its a numpy array with no columns)
X_data=X.values
Y_data=Y.values

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size = 0.2, random_state = 42)

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size = 0.2, random_state = 42)

#Building another model/classifier ISOLATION FOREST
from sklearn.ensemble import IsolationForest
ifc=IsolationForest(max_samples=len(X_train),
 contamination=outlier_fraction,random_state=1)
ifc.fit(X_train)
scores_pred = ifc.decision_function(X_train)
y_pred = ifc.predict(X_test)


# Reshape the prediction values to 0 for valid, 1 for fraud. 
y_pred[y_pred == 1] = 0
y_pred[y_pred == -1] = 1
n_errors = (y_pred != Y_test).sum()


#printing the confusion matrix
LABELS = [‘Normal’, ‘Fraud’]
conf_matrix = confusion_matrix(Y_test, y_pred)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS,
 yticklabels=LABELS, annot=True, fmt=”d”);
plt.title(“Confusion matrix”)
plt.ylabel(‘True class’)
plt.xlabel(‘Predicted class’)
plt.show()

#evaluation of the model
#printing every score of the classifier
#scoring in any thing
from sklearn.metrics import confusion_matrix
n_outliers = len(Fraud)
print(“the Model used is {}”.format(“Isolation Forest”))
acc= accuracy_score(Y_test,y_pred)
print(“The accuracy is {}”.format(acc))
prec= precision_score(Y_test,y_pred)
print(“The precision is {}”.format(prec))
rec= recall_score(Y_test,y_pred)
print(“The recall is {}”.format(rec))
f1= f1_score(Y_test,y_pred)
print(“The F1-Score is {}”.format(f1))
MCC=matthews_corrcoef(Y_test,y_pred)
print(“The Matthews correlation coefficient is{}”.format(MCC))
