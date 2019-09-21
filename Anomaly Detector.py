import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


trades=pd.read_csv('D:\Semester 7\Machine Learning\Assignments\Assignment 2\Trades_New.csv')
labels=trades.loc[trades['Stock']=='ES0158252033']
x=labels[['Sell Broker ID','Executed Qty']]
X=np.array(x)
#print(x.shape)
#trade.head(3)
#
x_trade=labels[['Sell Broker ID']]
y_trade=labels[['Executed Qty']]


y=pd.Categorical(x['Sell Broker ID'])

X_trade=np.array(x_trade)



categories=np.array(y)
values=np.array(y_trade['Executed Qty'])
valcats=np.array(x_trade['Sell Broker ID'])

f, u = pd.factorize(valcats)
q=np.bincount(f, values).astype(values.dtype)

Trd=zip(u,q)
Trd_set=set(Trd)
Trend_Data_Frame=pd.DataFrame(Trd_set);
#print(Trend_Data_Frame.index)

#plt.figure(figsize=(10,10))
#plt.scatter(Trend_Data_Frame.index,q,s=10)

c=zip(Trend_Data_Frame.index,Trend_Data_Frame[1])
T=pd.DataFrame(set(c))

#print(T[1])
##########################################################################

#----------------------K-Means Clustering-------------------------------#
#from sklearn.cluster import KMeans
#kmeans=KMeans(n_clusters=4) #value of k=4
#model=kmeans.fit(T)  #training
#y_kmeans=kmeans.predict(T)
#centers=model.cluster_centers_
#print(centers)


#plt.scatter(centers[:,0],centers[:,1],c='black',s=100,alpha=0.8);
##########################################################################


########################################################################

#--------------------Elbow Method--------------------#
#from sklearn.metrics import silhouette_score
# 
#print(silhouette_score(T,kmeans.labels_))
# 
# 
#from yellowbrick.cluster import KElbowVisualizer
# 
#mdl=KMeans(random_state=0)
# 
#visualizer=KElbowVisualizer(mdl,k=(2,10),metric='silhouette',timings=False)
# 
#visualizer.fit(T)
#visualizer.poof(T)

########################################################################


####################################################################################

fig=plt.figure(figsize=(15,15))

def plot_model(labels,alg_name,plot_index):
    
    ax=fig.add_subplot(3,2,plot_index)
    color_code={'anomaly':'red','normal':'green'}
    colors=[color_code[c] for c in labels]
    
    ax.scatter(Trend_Data_Frame.index,Trend_Data_Frame[1],color=colors,marker='.',label='red=anomaly')
    ax.legend(loc='upper right')
    
    leg=plt.gca().get_legend()
    leg.legendHandles[0].set_color('red')
    
    ax.set_title(alg_name)
    
#    
#    
#    
#    
outlier_fraction=0.05
Q=q.reshape(-1,1)

#########################################################################################

#-------------- DBSCAN-----------------------#
#from sklearn.cluster import DBSCAN

#model=DBSCAN(eps=0.63).fit(T)    
#labels_DB_S=model.labels_
#print(labels_DB_S)
#
#labels_DB_S=[('anomaly' if j==-1 else 'normal') for j in labels_DB_S]
#plot_model(labels_DB_S,'DBSCAN',2)

########################################################################################


#-----------Isolation Forest------------------#
#
#from sklearn.ensemble import IsolationForest
#
#from scipy import stats
#
#
#model=IsolationForest().fit(T)
#scores_pred=model.decision_function(T)
#threshold=stats.scoreatpercentile(scores_pred,100*outlier_fraction)
#
#
#labels_I_F=[('anomaly' if i<threshold else 'normal') for i in scores_pred]
##print(labels)
#
#plot_model(labels_I_F,'Isolation Forest',2)


###########################################################################################

#----------------LocalOutlierFactor-------------------------#

from sklearn.neighbors import LocalOutlierFactor
from scipy import stats


model=LocalOutlierFactor()
model.fit_predict(Q)
scores_pred=model.negative_outlier_factor_
threshold=stats.scoreatpercentile(scores_pred,100*outlier_fraction)

labels_L_O_F=[('anomaly' if i<threshold else 'normal') for i in scores_pred]
print(labels_L_O_F)

plot_model(labels_L_O_F,'LocalOutlierFactor',6)

#####################################################################################

























