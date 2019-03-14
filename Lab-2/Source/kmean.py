import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans

# loading collge dataseta
dataset = pd.read_csv('holiday.csv')

print(dataset.corr())
data_frame = dataset.iloc[:,[1,2,3,4,5,6]]
x=data_frame

#normalizing and preprocessing Data
scaler = preprocessing.StandardScaler()
scaler.fit(x)
X_scaled_array = scaler.transform(x)
X_scaled = pd.DataFrame(X_scaled_array, columns = x.columns)

nclusters = 3
seed = 0

km = KMeans(n_clusters=nclusters, random_state=seed)
km.fit(X_scaled)

# predict the cluster for each data point
y_cluster_kmeans = km.predict(X_scaled)

#silhouette_score
from sklearn import metrics
score = metrics.silhouette_score(X_scaled, y_cluster_kmeans)
print('silhouette_score :', score)

wcss =[]
#elbow method
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('Elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()


#Plotting Clusters
LABEL_COLOR_MAP = {0 : 'red',
                   1 : 'black',
                   2 : 'cyan',
                   3 : 'green',
                   4 : 'gold',
                   5 : 'blue',
                   6 : 'indigo',
                   7 : 'pink',
                   8 : 'lightblue',
                   9 : 'grey',
                   10: 'navy'
                   }
label_color = [LABEL_COLOR_MAP[l] for l in km.predict(X_scaled)]
plt.scatter(X_scaled_array[:, 0], X_scaled_array[:, 1], c=label_color)
plt.show()