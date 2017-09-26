import time
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

def read_csv(train_path, test_path):
    train_dataset = np.genfromtxt(train_path,delimiter=',',dtype=int)
    test_dataset = np.genfromtxt(test_path,delimiter=',',dtype=int)

    '''
        Parse images and labels from training and testing dataset
    '''
    labels_train = train_dataset[1:,0]
    images_train = train_dataset[1:,1:]

    labels_test = test_dataset[1:,0]
    images_test = test_dataset[1:,1:]
    
    return images_train, labels_train, images_test, labels_test
    
    
'''
    Calculates euclidean distance
'''
def euclideanDistance(data_1, data_2, num_feat):
    dist = 0
    for feat in range(num_feat):
        dist += pow((data_1[feat] - data_2[feat]), 2)
        
    return np.sqrt(dist)


'''
    Returns the indices of nearest neighbors
'''
def getNeighbors(data_1, data_2, k):
    row, col = data_1.shape
    
    distances = []
    for each_data in data_1:
        dist = euclideanDistance(each_data, data_2, col)
        distances.append(dist)
        
    nn_index = np.argsort(distances)[:k]
    
    return nn_index


'''
    Returns the highest predicted neighbor
'''
def getPrediction(neighbors, gt_labels):
    preds = []
    for index in neighbors:
        preds.append(gt_labels[index])
    total_preds = np.array(preds, dtype='int')
    total_bins = np.bincount(total_preds)
    highest_preds = np.argmax(total_bins)
    return highest_preds


'''
    Calculates prediction
'''
def knn_predict(images_train, labels_train, images_test, k):
    prediction_list = []
    for each_data in images_test:
        neighbors = getNeighbors(images_train, each_data, k)
        prediction = getPrediction(neighbors, labels_train)
        prediction_list.append(prediction)
    return np.array(prediction_list)


'''
    Calculates accuracy
'''
def getAccuracy(gt_labels, pred_labels):
    correct = 0
    if(gt_labels.size != pred_labels.size):
        print("Ground Truth and Prediction Labels vectors are not equal size")
        return
    else:
        total = pred_labels.size
        correct += (pred_labels == gt_labels).sum()
        return (100 * correct / total)

    
'''
    Calculates accuracy of each class
'''
def getAccuracyEachClass(gt_labels, pred_labels):
    each_class_acc = []
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    c = (pred_labels == gt_labels)
    for i in range(gt_labels.size):
        label = gt_labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1
        
    for i in range(10):
        acc = 100 * class_correct[i] / class_total[i]
        each_class_acc.append(acc)
        
    return each_class_acc


'''
    Returns accuracy of each class and overall accuracy
'''
def kNN(images_train, labels_train, images_test, labels_test, k):
    total_data = labels_train.size
    labels_pred = knn_predict(images_train, labels_train, images_test, k)
    acc = getAccuracyEachClass(labels_test, labels_pred)
    acc_av = getAccuracy(labels_test, labels_pred)

    return acc, acc_av


'''
    This function tests 1000 testing images on varying number of
    training data points which increases logrithmically from 30 
    to 10000 spaced by 10.
    
    Returns the average and each_class accuracy of kNN based on
    different values k on different dataset sizes
'''
def question_7c(k_list, dataset_list, images_train, labels_train, images_test, labels_test):
    total_k_acc_av = []
    total_k_acc = []

    for k in k_list:
        total_ds_acc = []
        total_ds_acc_av = []
        for ds in dataset_list:
            small_images_train = images_train[0:int(ds),0:]
            small_labels_train = labels_train[0:int(ds)]
            small_images_test = images_test[0:1000,0:]
            small_labels_test = labels_test[0:1000]
            
            since = time.time()
            acc, acc_av = kNN(small_images_train, small_labels_train, small_images_test, small_labels_test, k)
            time_taken = time.time() - since
                    
            total_ds_acc.append(acc)
            total_ds_acc_av.append(acc_av)
        
            print('**********Results of kNN with k = %d with %d datasize***************' % (k,ds))
            for i in range(10):
                print('Accuracy of %s : %.2f %%' % (classes[i], acc[i]))
            print('Overall accuray of kNN : %.2f %%' % acc_av)
            print('Time taken to test 1000 test images on %d dataset : %.2f minutes' % (ds, time_taken/60))
            print()
                  
        total_k_acc_av.append(total_ds_acc_av)
        total_k_acc.append(total_ds_acc)
        
    return total_k_acc, total_k_acc_av


'''
    This functions splits 2000 training dataset into
    training and validation and trains them.
    
    Returns the best k value and overall accuracy of
    kNN based on different values of k.
'''
def question_7e(k_list, dataset_list, images_train, labels_train):
    total_k_acc_av = []
    best_k = 0
    
    small_images_train = images_train[0:2000,0:]
    small_labels_train = labels_train[0:2000]
    
    for k in k_list:
        very_small_images_train = small_images_train[0:100,0:]
        very_small_labels_train = labels_train[0:100]

        very_small_images_val = small_images_train[100:200,0:]
        very_small_labels_val = labels_train[100:200]

        _, acc_av = kNN(very_small_images_train, very_small_labels_train, very_small_images_val, very_small_labels_val, k)
        
        if acc_av > best_k:
            best_k = acc_av

        total_k_acc_av.append(acc_av)

        print('**********Results of kNN when k = %d***************' % k)
        print('Overall accuray of kNN : %.2f %%' % acc_av)
        print()
        
    return best_k, total_k_acc_av

def main():
    print('Starting kNN implementation')
    # List of k
    #k_list = [1, 2, 3, 5, 10]
    k_list = [1,2]

    # List of logarithmic limit from 30 to 10000 for datapoints
    #dataset_list = np.logspace(1.477212,4,10).astype(dtype=int)
    dataset_list = np.logspace(1.477212,2,3).astype(dtype=int)

    # Read training/testing dataset
    print('Loading training and testing dataset')
    images_train, labels_train, images_test, labels_test = read_csv('../data/train.csv','../data/test.csv')

    print('Training in progress')
    total_k_acc, total_k_acc_av = question_7c(k_list, dataset_list, images_train, labels_train, images_test, labels_test)

    '''
        Display graph for question 7(c)
    '''
    fig1, ax1 = plt.subplots()
    ax1.plot(dataset_list, total_k_acc_av[0],'-o')
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(10))
    ax1.set_xlabel('Number of training data')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Accuracy of kNN on 10 different dataset when k = 1')
    plt.savefig('graph_7c')
    plt.close(fig1)
    
    '''
        Display graph for question 7(d)
    '''
    fig2, ax2 = plt.subplots()
    ax2.plot(dataset_list, total_k_acc_av[0], '-o')
    ax2.plot(dataset_list, total_k_acc_av[1], '-o')
    #ax2.plot(dataset_list, total_k_acc_av[2], '-o')
    #ax2.plot(dataset_list, total_k_acc_av[3], '-o')
    #ax2.plot(dataset_list, total_k_acc_av[4], '-o')
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(10))
    #ax2.legend(['k = 1', 'k = 2', 'k = 3', 'k = 5', 'k = 10'], loc='upper left')
    ax2.legend(['k = 1', 'k = 2'], loc='upper left')
    ax2.set_xlabel('Number of training data')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy of kNN varying with k')
    plt.savefig('graph_7d')
    plt.close(fig2)
    
    '''
        Display graph for question 7(e)
    '''
    best_k, acc_av_each_k = question_7e(k_list, dataset_list, images_train, labels_train)
    
    print('Best k : %d' % best_k)
    fig3, ax3 = plt.subplots()
    ax3.plot(k_list, acc_av_each_k, '-o')
    ax3.yaxis.set_major_locator(ticker.MaxNLocator(10))
    ax3.set_xlabel('k')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Accuracy of kNN on wrt k')
    plt.savefig('graph_7e')
    plt.close(fig3)

    
if __name__ == '__main__':
    main()
