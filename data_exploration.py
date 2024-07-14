
# import the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

# Load the dataset from the csv file using pandas
data = pd.read_csv(‘creditcard.csv’)

# data exploration, 
# premier regard de la donnée, à quoi elle ressemble
data.head()

# Print the shape of the data
# data = data.sample(frac=0.1, random_state = 48)
print(data.shape)
print(data.describe())

# distribution of anomalous features
features = data.iloc[:,0:28].columns
plt.figure(figsize=(12,28*4))
gs = gridspec.GridSpec(28, 1)
for i, c in enumerate(data[features]):
 ax = plt.subplot(gs[i])
 sns.distplot(data[c][data.Class == 1], bins=50)
 sns.distplot(data[c][data.Class == 0], bins=50)
 ax.set_xlabel(‘’)
 ax.set_title(‘histogram of feature: ‘ + str(c))
plt.show()

# Determine number of fraud cases in dataset
Fraud = data[data[‘Class’] == 1]
Valid = data[data[‘Class’] == 0]
outlier_fraction = len(Fraud)/float(len(Valid))
print(outlier_fraction)
print(‘Fraud Cases: {}’.format(len(data[data[‘Class’] == 1])))
print(‘Valid Transactions: {}’.format(len(data[data[‘Class’] == 0])))

# Correlation matrix
corrmat = data.corr()
fig = plt.figure(figsize = (12, 9))
sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()


#dividing the X and the Y from the dataset
X=data.drop([‘Class’], axis=1)
Y=data[“Class”]
print(X.shape)
print(Y.shape)
#getting just the values for the sake of processing (its a numpy array with no columns)
X_data=X.values
Y_data=Y.values


