'''
    class:      CS 674
    assignment: Homework #1
    name:       Won Cho
    email:      wcho3@gmu.edu
    date:       2/22/2017
'''

import numpy as np
import math
import random
import datetime
import matplotlib.pyplot as plt

#######################################################################
## method: loadData(dataset_path, test_size, window_size)    
#       loading training and testing data
##
## parameters:
##      dataset_path: a time series dataset file location
##      test_size: test sample size
##      window_size: subsequence size
## return:
##      X_train, y_train, X_test, y_test
#######################################################################
def loadData(dataset_path, test_size, window_size):

    ds_train = np.loadtxt(dataset_path + "/train.txt")
    ds_test = np.loadtxt(dataset_path + "/test.txt")
    X_train = nomalize(ds_train[:, 1:])
    y_train = ds_train[:, 0]
    samp_idxs = random.sample(range(ds_test.shape[0]),test_size)
    X_test = nomalize(ds_test[samp_idxs, 1:])
    y_test = ds_test[samp_idxs, 0]
    
    X_train = lower_dim_by_mean(X_train, window_size)
    X_test = lower_dim_by_mean(X_test, window_size)

    return X_train, y_train, X_test, y_test

#######################################################################
## method: nomalize(X) 
##      nomalizing
## parameters:
##      X: a time series dataset
## return:
##      X: nomalized dataset
#######################################################################
def nomalize(X):
    for i in range(X.shape[0]):
        X[i] = (X[i] - np.mean(X[i]))/np.std(X[i])
    return X

#######################################################################
## method: lower_dim_by_mean(X, window_size)
##      lowing dimentionality by using mean of each sub sequence
## parameters:
##      X: a time series dataset
##      window_size: subsequence size
## return:
##      X: reduced dimension from n to n/window_size
#######################################################################
def lower_dim_by_mean(X, window_size):
    n = X.shape[1]
    #m = int(p * n)
    m = window_size
    X2 = []
    for i in range(X.shape[0]):
        j = 0
        x = []
        while j < n:
            k = min(j + m, n)
            x.append(np.mean(X[i,j:k]))
            j = j + m
        X2.append(x)
    return np.asarray(X2)

#######################################################################
## method: euc_dist(x1, x2)
##      calculating euclidean distance of two time series
## parameters:
##      x1: a time series
##      x2: a time series
## return:
##      float: euclidean distance between x1 and x2
#######################################################################
def euc_dist(x1, x2):
    return np.sqrt(sum((x1 - x2)**2))

#######################################################################
## method: dtw_dist(x1, x2, w)
##      calculating DTW distance
##      if w = -1 then do unlimited w
##      else do do limited w
## parameters:
##      x1: a time series
##      x2: a time series
## return:
##      float:  dtw distance between x1 and x2
#######################################################################
def dtw_dist(x1, x2, w):
    n1 = len(x1)
    n2 = len(x2) 
    dtw = np.empty(shape = (n1, n2))

    if w == -1:
        for i in range(1, n1):
            dtw[i, 0] = float('inf')

        for j in range(1, n2):
            dtw[0, j] = float('inf')
        
    else:
        for i in range(0, n1):
            for j in range(0, n2):
                dtw[i, j] = float('inf')
        
    dtw[0,0] = 0

    for i in range(1, n1):
        if w == -1:
            for j in range (1, n2): 
                cost = (x1[i] - x2[j]) ** 2
                dtw[i,j] = cost + min(dtw[i - 1, j], dtw[i, j -1], dtw[i - 1, j - 1])

        else:
            for j in range (max(1, i -w), min(n2, i + w)):
                cost = (x1[i] - x2[j]) ** 2
                dtw[i,j] = cost + min(dtw[i - 1, j], dtw[i, j -1], dtw[i - 1, j - 1])
            

    return np.sqrt(dtw[n1 - 1, n2 - 1])

#######################################################################
## method: knn_1_dtw(X_train,X_test, y_train, w)
##      classify time series by using 1 nearest neighbor (1NN) and DTW
## parameters:
##      X_train: training time series set
##      X_test:  testing time series set
##      w: DTW w size
## return:
##      pred_y: predicted labels
##      t_sec: running time period
#######################################################################
def knn_1_dtw(X_train,X_test, y_train, w):
    t_start = datetime.datetime.now()
    pred_y = np.empty(shape = (X_test.shape[0]))
    pred_y.fill(-1)
    
    for i in range(X_test.shape[0]):
        min_j = -1
        min_dist = float('inf')
        for j in range(X_train.shape[0]):
            
            if lower_bound_keogh(X_test[i],X_train[j],5) < min_dist:
                dist = dtw_dist(X_test[i], X_train[j],w)
                if dist < min_dist:
                    min_dist = dist
                    min_j = j

        pred_y[i] = y_train[min_j]
    t_end = datetime.datetime.now()
    t_sec = (t_end - t_start).total_seconds()
    return pred_y, t_sec

#######################################################################
## method: knn_1_euc(X_train,X_test, y_train)
##      classify time series by using 1 nearest neighbor (1NN) and euclidean distance
## parameters:
##      X_train: training time series set
##      X_test:  testing time series set
## return:
##      pred_y: predicted labels
##      t_sec: running time period
#######################################################################
def knn_1_euc(X_train,X_test, y_train):
    t_start = datetime.datetime.now()
    pred_y = np.empty(shape = (X_test.shape[0]))
    pred_y.fill(-1)
    
    for i in range(X_test.shape[0]):
        min_j = -1
        min_dist = float('inf')
        for j in range(X_train.shape[0]):
            
            if lower_bound_keogh(X_test[i],X_train[j],5) < min_dist:
                dist = euc_dist(X_test[i], X_train[j])
                if dist < min_dist:
                    min_dist = dist
                    min_j = j

        pred_y[i] = y_train[min_j]
    t_end = datetime.datetime.now()
    t_sec = (t_end - t_start).total_seconds()
    return pred_y, t_sec

#######################################################################
## method: ower_bound_keogh(x1,x2,r)
##      lower bounding with keogh
## parameters:
##      x1: a time series
##      x2: a time series
##      r: boundary limit
## return:
##      float: lower bounding distance
#######################################################################
def lower_bound_keogh(x1,x2,r):
    dist_sum=0
    for ind,i in enumerate(x1):

        lower_bound=min(x2[(ind-r if ind-r>=0 else 0):(ind+r)])
        upper_bound=max(x2[(ind-r if ind-r>=0 else 0):(ind+r)])

        if i>upper_bound:
            dist_sum=dist_sum+(i-upper_bound)**2
        elif i<lower_bound:
            dist_sum=dist_sum+(i-lower_bound)**2

    return np.sqrt(dist_sum)

#######################################################################
## method: calc_accuracy(pred_y, y)
##      calculating accuracy
## parameters:
##      pred_y: predicted labels
##      y:      true labels
## return:
##      float: accuracy
#######################################################################
def calc_accuracy(pred_y, y):
    return sum(pred_y == y)/float(len(y))

#######################################################################
## report_rst1(datasets, RST):
##      accuracy and performace report by using different datasets
## parameters:
##      datasets: dataset name list
##      RST: result
#######################################################################
def report_rst1(datasets, RST):
    
    #accuracy report
    euc_acc = RST[:,0].tolist()
    dtw_unlimit_acc = RST[:,1].tolist()
    dtw_limit_acc = RST[:,2].tolist()
    ind = np.arange(len(datasets))
    width = 0.25
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, euc_acc, width, color='r' )
    rects2 = ax.bar(ind + width, dtw_unlimit_acc, width, color='y' )
    rects3 = ax.bar(ind + width + width, dtw_limit_acc, width, color='b' )

    ax.set_ylabel('accuracy')
    ax.set_title("1nn time series accuracy by dataset")
    ax.set_xticks(ind + width)
    ax.set_xticklabels(datasets)
    ax.legend((rects1[0], rects2[0], rects3[0]), ('euclidean dist', 'mtw unlimited dist', 'mtw limited dist'),loc='center left', bbox_to_anchor=(0.7, 0.5) )
    #plt.show()
    plt.savefig('knn_acc_dataset.png')
    #time report
    euc_time = RST[:,3].tolist()
    dtw_unlimit_time = RST[:,4].tolist()
    dtw_limit_time = RST[:,5].tolist()
    ind = np.arange(len(datasets))
    width = 0.25
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, euc_time, width, color='r' )
    rects2 = ax.bar(ind + width, dtw_unlimit_time, width, color='y' )
    rects3 = ax.bar(ind + width + width, dtw_limit_time, width, color='b' )

    ax.set_ylabel('time (sec)')
    ax.set_title("1nn time series performace by dataset")
    ax.set_xticks(ind + width)
    ax.set_xticklabels(datasets)
    ax.legend((rects1[0], rects2[0], rects3[0]), ('euclidean dist', 'mtw unlimited dist','mtw limited dist'),loc='center left', bbox_to_anchor=(0.5, 0.5) )
    #plt.show()
    plt.savefig('knn_time_dataset.png')

#######################################################################
## report_rst1(datasets, RST):
##      accuracy and performace report by using different warping(w) size
## parameters:
##      datasets: dataset name list
##      RST: result
#######################################################################
def report_rst2(datasets, RST):
    
    #accuracy report
    dtw_limit_acc = RST[:,0].tolist()
    ind = np.arange(len(datasets))
    #width = 0.25
    fig, ax = plt.subplots()
    rects1 = ax.plot(ind, dtw_limit_acc, color='r' )
    
    ax.set_ylabel('accuracy')
    ax.set_title("1nn time series accuracy by dtw w size")
    ax.set_xticks(ind)
    ax.set_xlabel('w size')
    #ax.set_xticklabels(datasets)
    #ax.legend(rects1[0], 'mtw limited dist',loc='center left', bbox_to_anchor=(0.7, 0.5) )
    #plt.show()
    plt.savefig("knn_acc_dtw_w.png")
    #time report
    dtw_limit_time = RST[:,1].tolist()
    ind = np.arange(len(datasets))
    #width = 0.25
    fig, ax = plt.subplots()
    rects1 = ax.plot(ind, dtw_limit_time,color='r' )
    
    ax.set_ylabel('time (sec)')
    ax.set_title("1nn time series performace by dtw w size")
    ax.set_xticks(ind)
    ax.set_xlabel('w size')
    #ax.set_xticklabels(datasets)
    #ax.legend((rects1[0]), ('mtw limited dist'),loc='center left', bbox_to_anchor=(0.5, 0.5) )
    #plt.show()
    plt.savefig("knn_time_dtw_w.png")

#######################################################################
## method: main()
##      main funtion
##      loading time series data
##      classifying time series by using 1NN and DTW
##      generating result report
#######################################################################
def main():
    """
    datasets = []
    RST1 = []  
    for i in range(1,6):
        #load data
        dataset_name = "dataset" + str(i) 
        X_train, y_train, X_test, y_test = loadData("hw1_datasets/" + dataset_name, test_size=30, window_size=1)
        #print "dataset", str(i), "dimension: ", X_train.shape, X_test.shape 

        #calculate knn
        pred_euc_y, euc_time = knn_1_euc(X_train, X_test, y_train)
        pred_dtw_unlimit_y, dtw_unlimit_time = knn_1_dtw(X_train, X_test, y_train, -1)
        pred_dtw_limit_y, dtw_limit_time = knn_1_dtw(X_train, X_test, y_train, 4)
        

        #calculate accuracy
        euc_accuracy = calc_accuracy(pred_euc_y, y_test)
        dtw_unlimit_accuracy = calc_accuracy(pred_dtw_unlimit_y, y_test)
        dtw_limit_accuracy = calc_accuracy(pred_dtw_limit_y, y_test)
        

        rst = [euc_accuracy, dtw_unlimit_accuracy, dtw_limit_accuracy, euc_time, dtw_unlimit_time, dtw_limit_time]

        RST1.append(rst)
        datasets.append(dataset_name)
        print "dataset", str(i), "performance: ", euc_accuracy, dtw_unlimit_accuracy, dtw_limit_accuracy, dtw_unlimit_time, dtw_limit_time
    print RST1
    
    #report_rst1(datasets, np.asarray(RST1))
    """

    RST2 = []
    w_names = []

    dataset_name = "dataset3"
    
    X_train, y_train, X_test, y_test = loadData("hw1_datasets/" + dataset_name, test_size=30, window_size=1)
    for i in range(1, 40):
        
        w_name = "w = " + str(i)
        #calculate knn
        #pred_euc_y, euc_time = knn_1_euc(X_train, X_test, y_train)
        #pred_dtw_unlimit_y, dtw_unlimit_time = knn_1_dtw(X_train, X_test, y_train, -1)
        pred_dtw_limit_y, dtw_limit_time = knn_1_dtw(X_train, X_test, y_train, w=i)
        

        #calculate accuracy
        #euc_accuracy = calc_accuracy(pred_euc_y, y_test)
        #dtw_unlimit_accuracy = calc_accuracy(pred_dtw_unlimit_y, y_test)
        dtw_limit_accuracy = calc_accuracy(pred_dtw_limit_y, y_test)
        

        rst = [dtw_limit_accuracy, dtw_limit_time]

        RST2.append(rst)
        w_names.append(w_name)
        print "w = ", str(i), "performance: ", dtw_limit_accuracy, dtw_limit_time
    
    print RST2

    #report_rst1(datasets, np.asarray(RST1))  
    report_rst2(w_names, np.asarray(RST2))

    

if __name__ == "__main__": main()