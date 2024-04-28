import os
import xml.etree.ElementTree as ET 
from PIL import Image
import numpy
import matplotlib.pyplot as plt




directory = os.getcwd()

anotts = 'annotations/Annotation'
images = 'images/Images'



def crop_images():

    save_image_folder = 'Processed_Images'

    anot_dir = os.path.join(directory, anotts)
    images_dir = os.path.join(directory, images)

    fail_count = 0

    for root, dirs, files in os.walk(anot_dir):
        folder_count = 0

        for folder in dirs:
            folder_path = os.path.join(root, folder)
            print(f"Processing folder: {folder}")
            
            file_count = 0
            try:
                for file in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file)
                    # Add your code to process the file here
                    with open(file_path, 'r') as f:
                        data = ET.fromstring(f.read())
                        bndbox = data.find('object').find('bndbox')

                        xmin = int(bndbox.find('xmin').text)
                        xmax = int(bndbox.find('xmax').text)
                        ymin = int(bndbox.find('ymin').text)
                        ymax = int(bndbox.find('ymax').text)


                    img_name = os.path.join(images_dir, folder, file.replace('xml', 'jpg')) + '.jpg'

                    img = Image.open(img_name)
                    img = img.crop((xmin, ymin, xmax, ymax))

                    save_image_name = str(folder_count) + '_' + file[10:] + '.jpg'
                    save_path = os.path.join(directory, save_image_folder, save_image_name)
                    img.save(save_path)
            except:
                print(f'{fail_count}: Failed on file:  {img_name}')




            folder_count += 1




def crop_image_square():
    save_image_folder = 'Processed_Images_Square'

    anot_dir = os.path.join(directory, anotts)
    images_dir = os.path.join(directory, images)

    fail_count = 0

    for root, dirs, files in os.walk(anot_dir):
        folder_count = 0

        for folder in dirs:
            folder_path = os.path.join(root, folder)
            print(f"Processing folder: {folder}")
            
            file_count = 0
            for file in os.listdir(folder_path):
                try:
                    file_path = os.path.join(folder_path, file)
                    # Add your code to process the file here
                    with open(file_path, 'r') as f:
                        data = ET.fromstring(f.read())
                        bndbox = data.find('object').find('bndbox')

                        xmin = int(bndbox.find('xmin').text)
                        xmax = int(bndbox.find('xmax').text)
                        ymin = int(bndbox.find('ymin').text)
                        ymax = int(bndbox.find('ymax').text)

                    use_x = True

                    if abs(xmax - xmin) < abs(ymax - ymin):
                        use_x = False

                    if use_x: 
                        img_size = abs(xmax - xmin)
                    else:
                        img_size = abs(ymax - ymin)

                    x_mid = (xmin + xmax) / 2
                    y_mid = (ymin + ymax) / 2

                    xmin = int(x_mid - img_size / 2)
                    xmax = int(x_mid + img_size / 2)
                    ymin = int(y_mid - img_size / 2)
                    ymax = int(y_mid + img_size / 2)

                    img_name = os.path.join(images_dir, folder, file.replace('xml', 'jpg')) + '.jpg'

                    img = Image.open(img_name)
                    img = img.crop((xmin, ymin, xmax, ymax))
                    img = img.resize((350, 350))

                    save_image_name = str(folder_count) + '_' + file[10:] + '.jpg'
                    if folder_count < 10:
                        save_image_name = '0' + save_image_name
                    save_path = os.path.join(directory, save_image_folder, save_image_name)
                    img.save(save_path)

                except:
                    print(f'{fail_count}: Failed on file:  {img_name}')
                    fail_count += 1
                    if fail_count == 10:
                        exit(0)




            folder_count += 1



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
        


if __name__ == '__main__':
    # crop_images()
    # crop_image_square()
    count_categories()