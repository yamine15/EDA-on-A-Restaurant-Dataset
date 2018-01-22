import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import preprocessing
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
#Read CSV and take a first look
df = pd.read_csv('~/data_08.csv')
df.info()
df.describe()
df.shape

#delete unecessary column
del df['Unnamed: 0']


#Check and transform datatype
df.dtypes
df['DateTime'] = pd.to_datetime(df['DateTime'])
df['Season'] = df['Season'].astype('category')
#Insert weekday column
df.insert(loc=1, column='Weekday', value=df['DateTime'].dt.weekday_name)

################################Start Data Cleaning############################

#Check missing value rows.
missing = df['Average.age.of.customers'].isnull()
df[df['Average.age.of.customers'].isnull()]
len(df[missing].index)/len(df)
#Since the missing data is only 2.9% of the dataset, and the reason behind the missing value is MAR, for the further analysis, we will delete it.
#drop missing value rows, where avg. age is nan, and reset index for comparison.
df = df.drop(df[missing].index)
#There are 11 cases that the revenues are not correspondent.
unusual = (df['Meal.revenue.in.EUR'] + df['Drink.revenue.in.EUR'] - df['Total.revenue.in.EUR']) > 10
df = df.drop(df[unusual].index)
#There are also 13 cases that the number of transactions and number of customers are not correspondent.
unusual1 = (df['No.transactions.with.card'] + df['No.transactions.with.cash'] != df['No.Customers'])
df = df.drop(df[unusual1].index)
df = df.reset_index(drop=True)
#Create df1, df2, df3 for the convenience of analysis
df0 = df.copy()
#Will be avarageTotal data
df1 = df0
#copy of original data without first two columns
df2 = df.loc[:, 'No.Customers':'Drink.revenue.in.EUR'].copy()
#copy of avarageTotal data without first two columns
df3 = df1.loc[:, 'No.Customers':'Drink.revenue.in.EUR'].copy()

df1['Tips.in.EUR'] = df1['Tips.in.EUR']/df1['No.Customers']
df1['No.times.door.opened'] = df1['No.times.door.opened']/df1['No.Customers']
df1['No.meals'] = df1['No.meals']/df1['No.Customers']
df1['No.drinks'] = df1['No.drinks']/df1['No.Customers']
df1['No.transactions.with.card'] = df1['No.transactions.with.card']/df1['No.Customers']
df1['No.transactions.with.cash'] = df1['No.transactions.with.cash']/df1['No.Customers']
df1['Meal.revenue.in.EUR'] = df1['Meal.revenue.in.EUR']/df1['Total.revenue.in.EUR']
df1['Drink.revenue.in.EUR'] = df1['Drink.revenue.in.EUR']/df1['Total.revenue.in.EUR']
df1['Total.revenue.in.EUR'] = df1['Total.revenue.in.EUR']/df1['No.Customers']


#check outliers
#Get the percentage of the outliers
Q1 = df1.quantile(0.25)
Q3 = df1.quantile(0.75)
IQR = Q3 - Q1
UpperLimit = Q3 + 1.5 * IQR
LowerLimit = Q1 - 1.5* IQR
((df1.loc[:, 'No.Customers':'Drink.revenue.in.EUR'] < (Q1 - 1.5*IQR)) | (df1.loc[:, 'No.Customers':'Drink.revenue.in.EUR'] > (Q3 + 1.5 * IQR))).sum()/len(df1)



#Get the boxplot of every variable start from No.Customers
for i in range(3, len(df1.columns)+1):
    fig = plt.figure()
    df1.boxplot(column=df1.columns[i])
    fig.savefig('~/Boxplot-' + df1.columns[i].replace(" ", "") + '.png')

#Find the unusual outliers by checking every box plot and store the indexes
outlierIndex = df1[(df1['No.drinks'] > 100) | (df1['No.meals'] > 20) |(df1['Total.revenue.in.EUR'] > 2000) | ((df1['Meal.revenue.in.EUR'] + df1['Drink.revenue.in.EUR']) < 0.97) |
        (df1['Tips.in.EUR'] > 12) ].index.values.tolist()

#Drop the extreme outliers first
df = df.drop(df.index[outlierIndex])
df0 = df0.drop(df0.index[outlierIndex])
df1 = df1.drop(df1.index[outlierIndex])
df2 = df2.drop(df2.index[outlierIndex])
df3 = df3.drop(df3.index[outlierIndex])

#Use Scatter plot to check pairs of variables and mark outliers for df2 
for i in range(0, len(df2.columns)+1):
    for j in range(0,i):
        fig = plt.figure()
        plt.scatter(df2[df2.columns[j]], df2[df2.columns[i]],alpha = 0.3,color = 'green')
        plt.scatter(df2[df2.columns[j]][df2[df2.columns[j]] > UpperLimit[j]], df2[df2.columns[i]][df2[df2.columns[j]]> UpperLimit[j]],alpha = 0.3,color = 'red')
        plt.scatter(df2[df2.columns[j]][df2[df2.columns[i]] > UpperLimit[i]], df2[df2.columns[i]][df2[df2.columns[i]]> UpperLimit[i]],alpha = 0.3,color = 'blue')
        plt.xlabel(df2.columns[j])
        plt.ylabel(df2.columns[i])
        fig.savefig('~' + df2.columns[j].replace(" ", "") + df2.columns[i].replace(" ", "") + '.png')

#Use Scatter plot to check pairs of variables and mark outliers for df3
df3 = df1.loc[:, 'No.Customers':'Drink.revenue.in.EUR'].copy()
for i in range(0, len(df3.columns)+1):
    for j in range(0,i):
        fig = plt.figure()
        plt.scatter(df3[df3.columns[j]], df3[df3.columns[i]],alpha = 0.3,color = 'green')
        
        plt.scatter(df3[df3.columns[j]][df3[df3.columns[j]] > UpperLimit[j]], df3[df3.columns[i]][df3[df3.columns[j]]> UpperLimit[j]],alpha = 0.3,color = 'red')
        plt.scatter(df3[df3.columns[j]][df3[df3.columns[i]] > UpperLimit[i]], df3[df3.columns[i]][df3[df3.columns[i]]> UpperLimit[i]],alpha = 0.3,color = 'blue')
        
        plt.scatter(df3[df3.columns[j]][df3[df3.columns[j]] < LowerLimit[j]], df3[df3.columns[i]][df3[df3.columns[j]]< LowerLimit[j]],alpha = 0.3,color = 'red')
        plt.scatter(df3[df3.columns[j]][df3[df3.columns[i]] < LowerLimit[i]], df3[df3.columns[i]][df3[df3.columns[i]]< LowerLimit[i]],alpha = 0.3,color = 'blue')
        
        plt.xlabel(df3.columns[j])
        plt.ylabel(df3.columns[i])
        fig.savefig('~/' + df3.columns[j].replace(" ", "") + df3.columns[i].replace(" ", "") + '.png')

#Get scaled data and check every column to get index, Final check of outliers
df3_scaled = preprocessing.StandardScaler().fit_transform(df3)
#One example to get index. 

((df3_scaled[:, 2] > 3.5) | (df3_scaled[:, 2] < -3.5)).tolist().index(True)

#Convert to dataframe for later use: PCA
df3_scaled = pd.DataFrame(df3_scaled, columns=df3.columns)

#After getting the index, we look into the observations and decided to keep them.
#Reindex
df = df.reset_index(drop=True)
df0 = df0.reset_index(drop=True)
df1 = df1.reset_index(drop=True)
df2 = df2.reset_index(drop=True)
df3 = df3.reset_index(drop=True)

df.to_csv('~/data.csv', index = False)
df0.to_csv('~/data_averageTotal.csv', index = False)


################################Data Cleaning Finished#########################


####################################Start Analysis#############################


#Perform the Pearson correlation test and generate heatmap
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(df.corr(method = 'pearson'), annot=True, linewidths=.5, fmt= '.3f',ax=ax)

#Perform the Spearman correlation test and generate heatmap
rho1, pval1 = spearmanr(df2)
pval1Truth = pval1 < 0.001
pval1Truth = pd.DataFrame(pval1Truth, index=df2.columns, columns=df2.columns)
spearCorMatrix = pd.DataFrame(rho1, index=df2.columns, columns=df2.columns)
spearCorMatrix = spearCorMatrix[pval1Truth]

f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(spearCorMatrix, annot=True, linewidths=.5, fmt= '.3f',ax=ax)



##############################Time Series Analysis#############################
#Start some , aggregate by month, weekday, etc, and plot them.


#Create a dataframe aggregating by seasons.
tss = df1.groupby('Season').mean()
tss = tss.reindex(["Winter", "Spring", "Summer", "Autumn"])
#Create time series plots aggregated by seasons.
for i in range(0, len(tss.columns)):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    tss[tss.columns[i]].plot(kind = 'bar', grid = True,figsize=(20, 10), color = 'g')
    ax1.set_xticks(np.arange(len(tss)))
    plt.xlabel('Season')
    plt.ylabel(tss.columns[i])
    fig.savefig('~/PerSeason/' + tss.columns[i].replace(" ", "") + 'PerSeason' + '.png')


#Create a dataframe aggregating by months.
ts = df1.set_index(['DateTime'])
tsm = ts.groupby(ts.index.month).mean()
#Create time series plots aggregated by months.
for i in range(0, len(tsm.columns)):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    tsm[tsm.columns[i]].plot(kind = 'line', grid = True,figsize=(20, 10), color = 'g')
    ax1.set_xticks(np.arange(len(tsm)))
    plt.xlabel('Month')
    plt.ylabel(tsm.columns[i])
    fig.savefig('~/PerMonth/' + tsm.columns[i].replace(" ", "") + 'PerMonth' + '.png')

#Create a dataframe aggregating by weekdays.
tsw = df1.groupby('Weekday').mean()
tsw = tsw.reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
#Create time series plots aggregated by weekdays.
for i in range(0, len(tsw.columns)):
    fig = plt.figure()
    tsw[tsw.columns[i]].plot(kind = 'line', grid = True,figsize=(20, 10), color = 'g')
    ax1.set_xticks(np.arange(len(tsw)))
    plt.xlabel('Weekday')
    plt.ylabel(tsw.columns[i])
    fig.savefig('~/PerWeekday/' + tsw.columns[i].replace(" ", "") + 'PerWeekday' + '.png')

#Create a dataframe aggregating by hours.
tsh = ts.groupby(ts.index.hour).mean()
#Create time series plots aggregated by hours.
for i in range(0, len(tsh.columns)):
    fig = plt.figure()
    tsh[tsh.columns[i]].plot(kind = 'line', grid = True,figsize=(20, 10), color = 'g')
    ax1.set_xticks(np.arange(len(tsh)))
    plt.xlabel('Hour')
    plt.ylabel(tsh.columns[i])
    fig.savefig('~/PerHour/' + tsh.columns[i].replace(" ", "") + 'PerHour' + '.png')


############################Time Series Analysis Finished######################

############################Start PCA for Clustering###########################
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(df3_scaled)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['pc1', 'pc2', 'pc3'])
pca.explained_variance_ratio_.sum()
fig = plt.figure()
plt.plot(pca.explained_variance_ratio_)
###################################PCA End#####################################

###############################Start Clustering################################
#Find the most promising number of k
SSE = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(df3_scaled)
    kmeans.inertia_
    SSE.append(kmeans.inertia_)

#Based on the graph, 4 is the probably the best choice for k
plt.plot(SSE)

kmeans = KMeans(n_clusters=4)
kmeans.fit(df3_scaled)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

df2.insert(loc=16, column='label', value=labels)

colors = ["g.","r.","y.","b."]
for i in range(len(principalComponents)):
    plt.plot(principalComponents[i][0], principalComponents[i][1],  colors[labels[i]], markersize = 2)
#Group data by labels.
gbl = df2.groupby('label').mean()

###############################Clustering Finished#############################


############################Some Interesting Findings##########################
