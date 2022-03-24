"""
Finished on Friday 27 March 2020

PROG-DS, Programming for Data Science (Spring 2020)

Project description:
This project is to plot the silhouette coefficient values against 
the number of clusters using the k mean clustering algorithm (Jaccard distance).

@author: Damon
@Email: 
"""

import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

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
Function "Precomputed_distances(data_arr)" is to compute distance matrix
input: a numpy array that need precomputed distances
return: distance matrix (used as precomputed distances)
"""
def Precomputed_distances(data_arr):                             
    arr_ndim = data_arr.shape[0]
    data_arr_DM = np.zeros([arr_ndim, arr_ndim])
    for i in range(arr_ndim):   
        data_arr_DM[i] = Jaccard_distance(data_arr[i], data_arr)
    return data_arr_DM

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
        k_index = (k_compare.sum(axis=0))// (k-1)                # Locate the indexes of vectors belonging to one cluster
        k_row_index = k_index.reshape(data_ndim, 1)              # Reshape, the values of index location are 1, others are 0
        k_num = k_index.sum()                                    # The total numbers of the vectors in one cluster
        if k_num == 0:                                           # The if statement is to avoid dividing by 0, keep the prototype unchanged
            k_arr[i] == k_arr[i]
        else:
            k_sum = (k_row_index * data_arr).sum(axis=0)         # The sum of all the vectors in one cluster
            k_mid = np.zeros(data_arr.shape[1]) + 0.5            # Creat a vector with all 0.5        
            k_arr[i] = ((k_sum/k_num) >= k_mid).astype(int)      # Get new prototype by calculating averages of all vectors in one cluster
        k_label += k_index * i                                   # Add points of the cluster assignment to the array
     
    return k_arr, k_label

"""
Function "Silhouette_coefficient" is to compute Silhouette coefficient of k mean clustering algorithm
input: the original data array or precomputed distances array (data_arr), the number of clusters (k)
return: Silhouette coefficient of the algrithom (Silhouette_Score)
"""
def Silhouette_coefficient(data, k):
                                    
    data_arr = data.copy()

    data_arr_radom = np.unique(data_arr.copy(), axis=0)          # Get an unique array to avoid same prototypes 
    np.random.shuffle(data_arr_radom)                            # Shuffle for a set of random prototypes
    
    k_arr = data_arr_radom[:k]                                   # Choose the first unique k vectors as prototypes
    k_arr_former = np.zeros(k_arr.shape)                         # Creat en all 0 array to store the former set of prototypes   
    k_label = np.zeros(data_arr.shape[0])                        # Creat en all 0 array to store the cluster assignment      
    Silhouette_Score = 0                                         # Initialize Silhouette score to zero
    
    # while statement is to implement iteration
    while (k_arr == k_arr_former).all() == False:                
            k_arr_former = k_arr.copy()                             
            k_arr, k_label= Jaccard_k_prototype(data_arr, k, k_arr)                  
    Silhouette_Score = metrics.silhouette_score(Precomputed_distances(data_arr), k_label, metric='precomputed')
    
    return Silhouette_Score

"""
Function "Silhouette_cluster_plot(file_path)" is to plot Silhouette coefficient
of k mean clustering algorithm against cluster numbers
input: the file path of the data (file_path)
return: the figure that plot Silhouette coefficient of  k mean clustering algorithm against cluster numbers
"""
def Silhouette_cluster_plot(file_path):

    data = pd.read_csv(file_path, sep=' ', 
                        skipinitialspace=True, header=None)      # Input data to DataFrame using pandas.read_csv 
    data = data.values                                           # Turn DataFrame to numpy array 
    plot_data = np.zeros((20, 1))                                # Set a all zero array to store Silhouette coefficient
    
    # The for statement is to compute the Silhouette coefficient when k is 2 to 20.
    for k in range(2,21):
        plot_data[k-2][0] = Silhouette_coefficient(data, k)
    
    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(2, 21), plot_data[:19, 0], 'bo', label='KMean-Jaccard' )
    ax.set_title('Silhouette Coefficients - dna_amp_chr_17.data')
    ax.legend(loc='best', shadow=True, fontsize='small')
    plt.xlabel('The Number of Clusters')
    plt.ylabel('Silhouette Coefficients')
    plt.xticks(np.arange(2, 21, step=1))
    plt.yticks(np.arange(11) / 10)
    plt.show()


file_path = input('Please enter the data file path: ')            # Input the data file path
Silhouette_cluster_plot(file_path)