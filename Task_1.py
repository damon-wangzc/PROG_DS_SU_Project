"""
Finished on Friday 27 March 2020

PROG-DS, Programming for Data Science (Spring 2020)

Project description:
This project is to implement a k-means clustering algorithm
, which uses Euclidean distance or Jaccard distance 
to compute similarity between the data and the cluster prototypes.

@author: Damon
@Email: 

"""

import numpy as np
import pandas as pd

"""
Function "Is_Binary(data_arr)" is a prejudgment function for data type
input: the data array that need to be judged (data_arr)
return: Boolean, if array is a binary one, return True; else False
"""
def Is_Binary(data_arr):
    binary_arr = np.array([0, 1])
    if np.unique(data_arr).shape[0] == 2:
        if (np.unique(data_arr) == binary_arr).all() == True:
            return True
    else:
        return False

"""
Function "Euclidean_distance(arr1, arr2)" is to compute Euclidean distance 
input: 2-D numpy arrays (continuous/numeric attribute) (arr1, arr2) 
       or one 2-D numpy array and one 1-D numpy array which has same size of axis 1 
return: a distance matrix 
"""
def Euclidean_distance(arr1, arr2):
    return np.sqrt(((arr1 - arr2) ** 2).sum(axis=1))

"""
Function "Jaccard_distance(arr1, arr2)" is to compute Jaccard distance 
input: 2-D numpy arrays (0-1 binary attribute) (arr1, arr2)
       or one 2-D numpy array and one 1-D numpy array which has same size of axis 1 
return: a distance matrix
"""
def Jaccard_distance(arr1, arr2):
    arr12 = (arr1 | arr2).sum(axis=1)                            
    arr12_zero = np.where(arr12 != 0, 1, arr12)                  # Mark 0 (the label of two all-0 arrays)
    arr12_nozero = np.where(arr12 == 0, 1, arr12)                # Replaced 0 with 1 (to avoid dividing by 0)
    distance = (1-(arr1 & arr2).sum(axis=1)/arr12_nozero) * arr12_zero
    return distance

"""
Function "Euclidean_k_prototype(data_arr, k, k_arr)" is to compute
to get a new set of prototypes and the new cluster assignment 
for Euclidean distance computing. 
input: orignal data array (data_arr), the number of clusters (k), prototypes(k_arr)
return: a new set of prototypes (k_arr) and the new cluster assignment (k_label)
"""
def Euclidean_k_prototype(data_arr, k, k_arr):
    
    data_ndim = data_arr.shape[0]                                # data_ndim means the number of repeating times or row number of data_arr
    k_label = np.zeros(data_ndim)                                # Create an all 0 array to store points of the cluster assignment 
    k_ED_arr = Euclidean_distance(k_arr[0], data_arr)            # Create en array to store the distances
    
    # The 1st for-loop is to get all the distance between protptypes 
    # and all vectors in data_arr, then store in k_ED_arr
    for i in range(1,k,1):
        k_ED = Euclidean_distance(k_arr[i], data_arr)            # Compute distances between other prototypes & all vectors in data_arr
        k_ED_arr = np.vstack((k_ED_arr, k_ED))                   # Add distances value to the array k_ED_arr

    # The for-loop is to assign points to k clusters 
    # by comparing arrays and get the new prototypes
    for i in range(k):
        k_compare = k_ED_arr[i] < k_ED_arr                       # Compare one prototype with the others
        k_index = (k_compare.sum(axis=0)) // (k-1)               # The indexes of vectors belonging to one cluster
        k_row_index = k_index.reshape(data_ndim, 1)              # Reshape, the values of indexe location are 1, others are 0
        k_num = k_index.sum()                                    # The total numbers of the vectors in one cluster
        if k_num == 0:                                           # The if statement is to avoid dividing by 0, keep the prototype unchanged
            k_arr[i] == k_arr[i]
        else:
            k_arr[i] = (k_row_index * data_arr).sum(axis=0) / k_num  # Get new prototype by calculating averages of all vectors in one cluster
        k_label += k_index * i                                    

    return k_arr, k_label

"""
Function "Jaccard_k_prototype(data_arr, k, k_arr)" is to compute
to get a new set of prototypes and the new cluster assignment 
for Jaccard distance computing. 
input: orignal data array (data_arr), the number of clusters (k), prototypes(k_arr)
return: a new set of prototypes (k_arr) and the new cluster assignment (k_label)
"""
def Jaccard_k_prototype(data_arr, k, k_arr):
    
    data_ndim = data_arr.shape[0]                                # data_ndim means the number of repeating times or row number of data_arr
    k_label = np.zeros(data_ndim)                                # Create an all 0 array to store points of the cluster assignment 
    k_JD_arr = Jaccard_distance(k_arr[0], data_arr)              # Create en array to store the distances

    # The 1st for-loop is to get all the distance values between each protptype and all vectors in data_arr
    # and store in an array k_ED_arr (k X data_ndim)
    for i in range(1, k, 1):
        k_JD = Jaccard_distance(k_arr[i], data_arr)              # Compute distances between other prototypes & all vectors in data_arr
        k_JD_arr = np.vstack((k_JD_arr, k_JD))                   # Add distances value to the array k_ED_arr

    # The if-statment is used to distinguish if the prototypes are slected in the data or computed
    # The for-loop is to assign points to k clusters by comparing arrays and get the new prototypes

    for i in range(k):
        k_compare = k_JD_arr[i] < k_JD_arr                       # Compare one prototype with the others
        k_index = (k_compare.sum(axis=0)) // (k-1)               # Locate the indexes of vectors belonging to one cluster
        k_row_index = k_index.reshape(data_ndim, 1)              # Reshape, the values of index location are 1, others are 0
        k_num = k_index.sum()                                    # The total numbers of the vectors in one cluster
        k_sum = (k_row_index * data_arr).sum(axis=0)             # The sum of all the vectors in one cluster
        k_mid = np.zeros(data_arr.shape[1]) + 0.5                # Creat a vector with all 0.5        
        k_arr[i] = ((k_sum/k_num) >= k_mid).astype(int)          # Get new prototype by calculating averages of all vectors in one cluster
        k_label += k_index * i                                   # Add points of the cluster assignment to the array
     
    return k_arr, k_label

"""
Function "KMean_Euclidean_Jaccard(file_path, k)" is the main function 
to implement k mean algorithm
input: the file path of the data (file_path), the number of clusters (k)
return: the final set of prototypes (cluster centers) and the cluster assignment
"""
def KMean_Euclidean_Jaccard(file_path, k):

    data = pd.read_csv(file_path, sep=' ', skipinitialspace=True, header=None)
    data = data.values                                           # Turn DataFrame to numpy array
    
    if Is_Binary(data):                                          # Judge the type of data                                     
        data_arr = data.copy()
    else:
        data_arr = (data - np.average(data, axis=0))/np.std(data, axis=0)    # normalize the data to have zero mean and unit variance
    
    data_arr_radom = np.unique(data_arr.copy(), axis=0)          # Get an unique array to avoid same prototypes
    np.random.shuffle(data_arr_radom)                            # Shuffle for a set of random prototypes
    
    k_arr = data_arr_radom[:k]                                   # Choose the first unique k vectors as prototypes
    k_arr_former = np.zeros(k_arr.shape)                         # Creat en all 0 array to store the former set of prototypes
    k_label = np.zeros(data_arr.shape[0])                        # Creat en all 0 array to store the cluster assignment
    k_label_former = np.full(data_arr.shape[0], 1)               # Creat an all 1 array as an initial label
    
    # The if statement is to dispath different functions and while statement is to implement iteration
    if Is_Binary(data):
        while (k_arr == k_arr_former).all() == False:
            k_arr_former = k_arr.copy()
            k_label_former = k_label.copy()
            k_arr, k_label= Jaccard_k_prototype(data_arr, k, k_arr)
    else:
        while (k_arr == k_arr_former).all() == False:
            k_arr_former = k_arr.copy()
            k_label_former = k_label.copy()
            k_arr, k_label= Euclidean_k_prototype(data_arr, k, k_arr)
        k_arr = k_arr * np.std(data, axis=0) + np.average(data, axis=0)
    
    return k_arr, k_label


file_path = input('Please enter the data file path: ')            # Input the data file path C:\Users\wangz\OneDrive\Desktop\prog-ds-project\measurements.data
k = int(input('Please enter the number of clusters(k value): '))  # Input the number of clusters

k_arr, k_label = KMean_Euclidean_Jaccard(file_path, k)
print(f'The {k} cluster centers are:' )
print(k_arr)                                                      # Print k cluster centers
print("The cluster assignment is ")
print(k_label)                                                    # Print the cluster assignment