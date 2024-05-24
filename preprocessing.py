import os
import xml.etree.ElementTree as ET 
from PIL import Image
import numpy
import matplotlib.pyplot as plt
import csv




directory = os.getcwd()



def count_categories():
    images_dir = os.path.join(directory, 'Processed_Images_Square')

    categories = {}

    for root, dirs, files in os.walk(images_dir):
        for file in files:

            cat_num = int(file.split('_')[0])

            if cat_num not in categories:
                categories[cat_num] = 1
            else:
                categories[cat_num] += 1


    y = numpy.zeros(len(categories))

    for x in categories:
        y[x] = categories[x]

    x = list(range(len(y)))



    fig = plt.figure(figsize = (10, 5))
 
    # creating the bar plot
    plt.bar(x, y, color ='maroon')
    
    plt.xlabel("Category Number")
    plt.ylabel("No. of photos in each category")
    plt.title("Distribution of categories")
    plt.show()
        


def dog_num_to_name(num):

    with open('dog_num_to_names.txt', 'r') as f:
        c = csv.reader(f)

        for row in c:
            if int(row[0]) == int(num):
                return row[1]
            

    return None



import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt     # for plotting
import numpy as np                  # for reshaping array manipulation
import cv2                          # for image loading and colour conversion
import tensorflow as tf             # for bulk image resize
import glob
import random
import keras
from sklearn import discriminant_analysis
from sklearn import decomposition
from sklearn.decomposition import IncrementalPCA
import numpy
from sklearn.manifold import TSNE
import tensorflow as tf             # for bulk image resize
import keras

from keras.layers import Dense, Conv2D, BatchNormalization, Activation, AveragePooling2D, Input, Flatten, MaxPool2D, SpatialDropout2D
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import seaborn as sns
import random
import copy



path = './Processed_Images_Square/'

image_dimensions = (350,350,3)


def gen_files(batch_size):
    files = glob.glob(path + '*.jpg')

    while True:
        x = []
        y = []
        for i in range(batch_size):
            f = random.choice(files)
            img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) / 255.0
            img = cv2.resize(img, (image_dimensions[0], image_dimensions[1]))
            x.append(img)
            f = f.split('/')[-1]
            dog_num = float(f.split('_')[0])
            y.append(dog_num)

        yield np.array(x), np.array(y)


def gen_files_vec(batch_size):
    image_generator = gen_files(batch_size)

    while True:
        x, y = next(image_generator)


        yield vectorise(x), y

def vectorise(images):
    # use numpy's reshape to vectorise the data
    return np.reshape(images, [len(images), -1])


def plot_images(images, labels):
    fig = plt.figure(figsize=[15, 18])
    loop_count = 50
    if len(images) < 50:
        loop_count = len(images)
    for i in range(loop_count):
        ax = fig.add_subplot(8, 6, i + 1)
        ax.imshow(images[i,:], cmap=plt.get_cmap('Greys'))
        ax.set_title(labels[i])
        ax.axis('off')


def get_siamese_data(batch_size):

    while True:
        files = glob.glob(path + '*.jpg')

        images = []
        labels = []

        for i in range(int(batch_size / 2)):
            ## Get original dog

            f = random.choice(files)
            original_dog = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) / 255.0
            
            
            ## Get matching pair
            f = f.split('/')[-1]
            dog_num = f.split('_')[0]
            
            matching_breed_files = glob.glob(path + dog_num + '*.jpg')
            matching_breed_files_without_original = copy.deepcopy(matching_breed_files)
            matching_breed_files_without_original.remove(path + f)

            matching_dog = random.choice(matching_breed_files)
            pair_dog = cv2.cvtColor(cv2.imread(matching_dog), cv2.COLOR_BGR2RGB) / 255.0

            images.append((original_dog, pair_dog))
            labels.append(1.0)

            ## Get non-matching pair
            non_matching_breed_files = [x for x in files if x not in matching_breed_files]

            non_matching_dog = random.choice(non_matching_breed_files)
            non_pair_dog = cv2.cvtColor(cv2.imread(non_matching_dog), cv2.COLOR_BGR2RGB) / 255.0
            
            images.append((original_dog, non_pair_dog))
            labels.append(0.0)
            

        yield np.array(images), np.array(labels)


def plot_pairs(x, y):
    fig = plt.figure(figsize=[25, 6])
    for i in range(10):
        ax = fig.add_subplot(2, 10, i*2 + 1)
        ax.imshow(x[i][0,:])
        ax.set_title('Pair ' + str(i) +'; Label: ' + str(y[i]))

        ax = fig.add_subplot(2, 10, i*2 + 2)
        ax.imshow(x[i][1,:])    
        ax.set_title('Pair ' + str(i) +'; Label: ' + str(y[i]))



def plot_tsne(data_x, data_y):
    tsne_embeddings = TSNE(random_state=4).fit_transform(data_x)
    fig = plt.figure(figsize=[12, 12])
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(tsne_embeddings[:,0], tsne_embeddings[:,1], c = data_y.flatten());


def cmc_to_top(cmc, verbose=True):
    top1 = cmc[0]
    top5 = cmc[4]
    top10 = cmc[9]

    if verbose:
        print(f'Top 1: {top1}')
        print(f'Top 5: {top5}')
        print(f'Top 10: {top10}')

    return top1, top5, top10
    

if __name__ == '__main__':
    # count_categories()
    print(count_categories())