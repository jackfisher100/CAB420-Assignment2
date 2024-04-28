# CAB420-Assignment2 #

## File name format ##

<category_number>_<filename>.jpg

e.g. file "00_1271.jpg" is a dog of the category 00 which happens to be a chihuahua. And image number 1271.

To get the category number in python 

~~~python

    directory = os.getcwd()
    images_dir = os.path.join(directory, 'Processed_Images_Square')
    
    for root, dirs, files in os.walk(images_dir):
        for file in files:
    
            # cat_num is the number representing that category
            cat_num = int(file.split('_')[0])

~~~
