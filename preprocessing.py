#hello
#woohoo


# finally

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





if __name__ == '__main__':
    # count_categories()
    print(count_categories())