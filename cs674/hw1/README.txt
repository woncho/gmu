#######################################################################
##  class:      CS 674
##  assignment: Homework #1
##  name:       Won Cho
##  email:      wcho3@gmu.edu
##  date:       2/22/2017
##
##  language:   python 2.7
##  dataset:    5 different time series datasets given by class     
#######################################################################

#######################################################################
## importing python libries
#######################################################################
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

#######################################################################
## method: nomalize(X) 
##      nomalizing
## parameters:
##      X: a time series dataset
## return:
##      X: nomalized dataset
#######################################################################
def nomalize(X):

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

#######################################################################
## report_rst1(datasets, RST):
##      accuracy and performace report by using different datasets
## parameters:
##      datasets: dataset name list
##      RST: result
#######################################################################
def report_rst1(datasets, RST):
    

#######################################################################
## report_rst1(datasets, RST):
##      accuracy and performace report by using different warping(w) size
## parameters:
##      datasets: dataset name list
##      RST: result
#######################################################################
def report_rst2(datasets, RST):
    

#######################################################################
## examle
#######################################################################    
datasets = []
RST1 = []  
#loading different time series dataset
for i in range(1,6):
    #load data
    dataset_name = "dataset" + str(i) 
    X_train, y_train, X_test, y_test = loadData("hw1_datasets/" + dataset_name, test_size=30, window_size=20)
    #print "dataset", str(i), "dimension: ", X_train.shape, X_test.shape 

    #calculate knn
    pred_euc_y, euc_time = knn_1_euc(X_train, X_test, y_train)
    pred_dtw_unlimit_y, dtw_unlimit_time = knn_1_dtw(X_train, X_test, y_train, -1)
    pred_dtw_limit_y, dtw_limit_time = knn_1_dtw(X_train, X_test, y_train, 4)
    

    #calculate accuracy
    euc_accuracy = calc_accuracy(pred_euc_y, y_test)
    dtw_unlimit_accuracy = calc_accuracy(pred_dtw_unlimit_y, y_test)
    dtw_limit_accuracy = calc_accuracy(pred_dtw_limit_y, y_test)
    
    #adding result
    rst = [euc_accuracy, dtw_unlimit_accuracy, dtw_limit_accuracy, euc_time, dtw_unlimit_time, dtw_limit_time]
    RST1.append(rst)
    datasets.append(dataset_name)
    print "dataset", str(i), "performance: ", euc_accuracy, dtw_unlimit_accuracy, dtw_limit_accuracy, dtw_unlimit_time, dtw_limit_time
print RST1


RST2 = []
w_names = []

dataset_name = "dataset3"

X_train, y_train, X_test, y_test = loadData("hw1_datasets/" + dataset_name, test_size=30, window_size=20)
# looping by DTW w size
for i in range(1, 10):
    
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

report_rst1(datasets, np.asarray(RST1))  
report_rst2(w_names, np.asarray(RST2))
