import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn import metrics

#importing the dataset
dataset = pd.read_csv('College.csv')

#performing clustering based on these features
x = dataset.iloc[:,[4, 18]]

#scaling the data using MinMaxScaler
scaler = preprocessing.MinMaxScaler()
scaler.fit(x)
X_scaled_array = scaler.transform(x)
X_scaled = pd.DataFrame(X_scaled_array, columns = x.columns)
#print(X_scaled)

#kmeans cluster
km = KMeans(n_clusters=4)
km.fit(X_scaled)

# predict the cluster for each data point
y_cluster_kmeans = km.predict(X_scaled)

#silhouette score
score = metrics.silhouette_score(X_scaled, y_cluster_kmeans)
print('silhouette_score :', score)

#initialising colors for the plot
LABEL_COLOR_MAP = {0 : 'r',
                   1 : 'g',
                   2 : 'k',
                   3 : 'b',
                   }

#assigning colors to the clustered points
label_color = [LABEL_COLOR_MAP[l] for l in km.predict(X_scaled)]
plt.figure(figsize=(8, 6))
plt.xlabel('enrollment')
plt.ylabel('Grad rate')
plt.title('K means clustering')
plt.scatter(X_scaled_array[:,0], X_scaled_array[:,1], c=label_color)
#print(X_scaled_array)
plt.show()
