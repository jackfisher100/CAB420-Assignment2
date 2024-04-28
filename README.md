# CAB420-Assignment2 #

## File name format ##

< category_number >_< fileName >.jpg

e.g. file "00_1271.jpg" is a dog of the category 00 which happens to be a chihuahua. And image number 1271.

To get the category number in python 

~~~python

directory = os.getcwd()
images_dir = os.path.join(directory, 'Processed_Images_Square')

for root, dirs, files in os.walk(images_dir):
    for file in files:

        # dog_num is the number representing that category
        dog_num = int(file.split('_')[0])
~~~


## Get the dog name from cat_num ##

~~~python

import preprocessing

# dog_num can be an int or string
dog_name = dog_num_to_name(dog_num)

print(dog_name)
~~~


## Show graph with the distribution of the categories ##
~~~python
import preprocessing

# If you want to use this graph in the notebook then just copy the code from that function
count_categories()

~~~