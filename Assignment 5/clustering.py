#-------------------------------------------------------------------------
# AUTHOR: Harry Nguyen   
# FILENAME: clustering.py
# SPECIFICATION: Cluster the data.
# i. Run K-means multiple times to find k that maximizes the Silhouette coefficient.
# ii. Plot the values of k and the corresponding Silhouette coefficients to visualize and confirm the best k found
# iii. Calculate and print the Homogeneity score of the best k clustering task using the test data file
# FOR: CS 4210- Assignment #5
# TIME SPENT: 35 minutes
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics

df = pd.read_csv('training_data.csv', sep=',', header=None) #reading the data by using Pandas library

#assign your training data to X_training feature matrix
X_training = np.array(df.values)

#run kmeans testing different k values from 2 until 20 clusters
     #Use:  kmeans = KMeans(n_clusters=k, random_state=0)
     #      kmeans.fit(X_training)
     #--> add your Python code
silhouette_coefficient_lst = []
for k in range(2, 21):
     kmeans = KMeans(n_clusters=k, random_state=0)
     kmeans.fit(X_training)

     #for each k, calculate the silhouette_coefficient by using: silhouette_score(X_training, kmeans.labels_)
     #find which k maximizes the silhouette_coefficient
     #--> add your Python code here
     silhouette_coefficient = silhouette_score(X_training, kmeans.labels_)
     silhouette_coefficient_lst.append(silhouette_coefficient)

#plot the value of the silhouette_coefficient for each k value of kmeans so that we can see the best k
#--> add your Python code here
k_lst = [k for k in range(2, 21)]
plt.plot(k_lst, silhouette_coefficient_lst)
plt.xlabel('k value')
plt.ylabel('Silhouette Coefficient')
plt.xticks(k_lst)
plt.grid()
plt.show()

#reading the test data (clusters) by using Pandas library
#--> add your Python code here
testing = pd.read_csv('testing_data.csv', header=None)

#assign your data labels to vector labels (you might need to reshape the row vector to a column vector)
# do this: np.array(df.values).reshape(1,<number of samples>)[0]
#--> add your Python code here
# -1 here means unknown dimension which will be figured out by numpy to make the new shape compatible with the original shape
# -1 will be replaced with the number of samples
labels = np.array(testing.values).reshape(1, -1)[0]

#Calculate and print the Homogeneity of this kmeans clustering
#Template: print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())
#--> add your Python code here
# Get the best k value which has the highest silhouette coefficient
max_silhouette = max(silhouette_coefficient_lst)
print('The highest silhouette coefficient we have is ' + str(max_silhouette))
best_k = k_lst[silhouette_coefficient_lst.index(max_silhouette)]
print('The best k value is ' + str(best_k))

kmeans = KMeans(n_clusters=best_k, random_state=0)
kmeans.fit(X_training)

#Calculate and print the Homogeneity of this kmeans clustering using the best k
print("K-Means Homogeneity Score using the best k value is " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())
