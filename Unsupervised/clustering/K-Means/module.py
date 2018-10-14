'''import csv

import csv
with open("wine_data.csv") as f:
    data = [r for r in csv.reader(f)]

header = [
    "Variety","Alcohol","Malic acid","Ash","Alcalinity of ash","Magnesium","Total phenols","Flavanoids","Nonflavanoid phenols",
    "Proanthocyanins","Color intensity","Hue","OD280/OD315 of diluted wines","Proline"
]'''

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi
import matplotlib.pyplot as plt

red = pd.read_csv('winequality-red.csv', low_memory=False, sep=';')
white = pd.read_csv('winequality-white.csv', low_memory=False, sep=';')


def call(functionToCall):
    print('Red')
    functionToCall(red)
    print('\n')

    print('White')
    functionToCall(white)
    print('\n')


# ----- to remove all spaces from column names ---------
def remove_col_spaces(wine_set):
    wine_set.columns = [x.strip().replace(' ', '_') for x in wine_set.columns]
    return wine_set

call(remove_col_spaces)


# ______________________________K-Means Cluster Analysis_________________

def k_means(wine_set):

    # standardize predictors to have mean=0 and sd=1
    pred = wine_set[["density", 'alcohol', 'sulphates', 'pH', 'volatile_acidity', 'chlorides', 'fixed_acidity',
                    'citric_acid', 'residual_sugar', 'free_sulfur_dioxide', 'total_sulfur_dioxide']]

    clustervar = pred.copy()

    clustervar = pd.DataFrame(preprocessing.scale(clustervar))
    clustervar.columns = pred.columns

    # split into training and testing sets
    clus_train, clus_test = train_test_split(clustervar, test_size=.3, random_state=123)
    # print(clus_train.shape)

    # _________________k-means cluster analysis for 1-9 clusters
    clusters = range(1, 10)
    meandist = []

    for k in clusters:
        # print(k)
        model = KMeans(n_clusters=k)
        model.fit(clus_train)
        # clusassign = model.predict(clus_train)
        meandist.append(sum(np.min(cdist(clus_train, model.cluster_centers_, 'euclidean'), axis=1))/clus_train.shape[0])

    print('Average distance from observations to the cluster centroids for 1-9 clusters:')
    print(meandist)

    # plot average distance from observations to the cluster centroid
    # to use the Elbow Method to identify number of clusters to choose
    plt.plot(clusters, meandist)
    plt.xlabel('Number of clusters')
    plt.ylabel('Average distance')
    plt.title('Selecting k with the Elbow Method')
    plt.show()

    ## Plot solution for each number of clusters (1-9) to choose the best
    for k in clusters:
        modelk = KMeans(n_clusters=k)
        modelk.fit(clus_train)
        # clusassign = modelk.predict(clus_train)
        #     # plot clusters
        pca_2 = PCA(2)
        plot_columns = pca_2.fit_transform(clus_train)
        plt.scatter(x=plot_columns[:, 0], y=plot_columns[:, 1], c=modelk.labels_)
        plt.xlabel('Canonical variable 1')
        plt.ylabel('Canonical variable 2')
        plt.title('Canonical variables for', k, 'clusters')
        plt.show()


    # 2-cluster solution proven to be the best
    model2 = KMeans(n_clusters=2)
    model2.fit(clus_train)
    # plot cluster
    pca_2 = PCA(2)
    plot_columns = pca_2.fit_transform(clus_train)
    plt.scatter(x=plot_columns[:, 0], y=plot_columns[:, 1], c=model2.labels_)
    plt.xlabel('Canonical variable 1')
    plt.ylabel('Canonical variable 2')
    plt.title('Canonical variables for 2 clusters')
    plt.show()

    # merge cluster assignment with clustering variables to examine cluster variable means by cluster

    # create a unique identifier variable from the index for the cluster training data
    # to merge with the cluster assignment variable

    clus_train.reset_index(level=0, inplace=True)
    # create a list that has the new index variable
    cluslist = list(clus_train['index'])
    # print(cluslist)
    # create a list of cluster assignments
    labels = list(model2.labels_)
    # combine index variable list with cluster assignment list into a dictionary
    newlist = dict(zip(cluslist, labels))
    # convert newlist dictionary to a dataframe
    newclus = pd.DataFrame.from_dict(newlist, orient='index')
    # rename the cluster assignment column
    newclus.columns = ['cluster']
    # create a unique identifier variable from the index for the cluster assignment dataframe
    # to merge with cluster training data
    newclus.reset_index(level=0, inplace=True)
    # merge the cluster assignment dataframe with the cluster training variable dataframe by the index variable
    merged_train = pd.merge(clus_train, newclus, on='index')
    # print(merged_train.head(n=100))
    print('\nCounts of observations per each cluster:')
    print(merged_train.cluster.value_counts())

    # calculate clustering variable means by cluster
    clustergrp = merged_train.groupby('cluster').mean()
    print('\nClustering variable means by cluster:')
    print(clustergrp)

    # validate clusters in training data by examining cluster differences
    # in wine quality (validation variable) using ANOVA
    # merge wine quality with clustering variables and cluster assignment data
    qual = wine_set['quality']
    # split quality data into train and test sets
    qual_train, qual_test = train_test_split(qual, test_size=.3, random_state=123)
    qual_train1 = pd.DataFrame(qual_train)
    qual_train1.reset_index(level=0, inplace=True)
    merged_train_all = pd.merge(qual_train1, merged_train, on='index')
    sub1 = merged_train_all[['quality', 'cluster']]

    mod = smf.ols(formula='quality ~ C(cluster)', data=sub1).fit()
    print(mod.summary())

    print('\nMeans for wine quality by cluster:')
    print(sub1.groupby('cluster').mean())
    print('\nStandard deviations for wine quality by cluster:')
    print(sub1.groupby('cluster').std())

    # perform Post hoc test (using Tukey's Honestly Significant Difference Test)
    mc1 = multi.MultiComparison(sub1['quality'], sub1['cluster'])
    res1 = mc1.tukeyhsd()
    print(res1.summary())

print('----------------K-Means Cluster Analysis------------------------')
call(k_means)