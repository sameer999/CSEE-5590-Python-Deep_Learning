from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import silhouette_score


sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")

dataset = pd.read_csv('College.csv')

x = dataset.iloc[:,[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]]
y = dataset.iloc[:,[1,18]]
#print(y.dtypes)

# see how many samples we have of each species
print(dataset["Private"].value_counts())

#print(dataset)

sns.FacetGrid(dataset, hue="Private", size=4).map(plt.scatter ,"Apps", "Accept")
# do same for petals
sns.FacetGrid(dataset, hue="name", size=4).map(plt.scatter, "Accept", "PhD")
sns.FacetGrid(dataset, hue="Private", size=4).map(plt.scatter, "Enroll","PhD")
plt.show()

scaler = preprocessing.StandardScaler()

scaler.fit(x)
X_scaled_array = scaler.transform(x)
X_scaled = pd.DataFrame(X_scaled_array, columns = x.columns)


km = KMeans(n_clusters=3, random_state=0)
km.fit(X_scaled)

# predict the cluster for each data point
y_cluster_kmeans = km.predict(X_scaled)

score = metrics.silhouette_score(X_scaled, y_cluster_kmeans)
print(score)


wcss = []
##elbow method to know the number of clusters
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()