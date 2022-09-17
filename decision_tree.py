#-------------------------------------------------------------------------
# AUTHOR: Harry Nguyen
# FILENAME: decision_tree
# SPECIFICATION: Get the dataset from a csv file, transfrom the categorical values in the dataset to numbers and use them
# to create and plot a decision tree
# FOR: CS 4210- Assignment #1
# TIME SPENT: 15 min
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv
db = []
X = []
Y = []

#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)
         print(row)

#transform the original categorical training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
# so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
for row in db:
    temp = []
    # Young = 1, Prepresbyopic = 2, Presbyopic = 3
    if row[0] == 'Young':
        temp.append(1)
    elif row[0] == 'Prepresbyopic':
        temp.append(2)
    else:
        temp.append(3)

    # Myope = 1, Hypermetrope = 2
    if row[1] == 'Myope':
        temp.append(1)
    else:
        temp.append(2)

    # Astigmatis Yes = 1, No = 2
    if row[2] == 'Yes':
        temp.append(1)
    else:
        temp.append(2)

    # Tear Production Normal = 1, Reduced = 2
    if row[3] == 'Normal':
        temp.append(1)
    else:
        temp.append(2)
        
    X.append(temp)

#transform the original categorical training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
for row in db:
    # Yes = 1, No = 2
    if row[-1] == 'Yes':
        Y.append(1)
    else:
        Y.append(2)

#fitting the decision tree to the data
clf = tree.DecisionTreeClassifier(criterion = 'entropy', )
clf = clf.fit(X, Y)

#plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()