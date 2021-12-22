from matplotlib import pyplot as plt
import os

img_dir = './dog_and_cat'
dog = []
cat = []
error = []
for i in os.listdir(img_dir + '/Dog/'):
    try:
        plt.imread(img_dir + '/Dog/' + i)
    except:
        error.append(img_dir + '/Dog/' + i)
        print(img_dir + '/Dog/' + i)
for i in os.listdir(img_dir + '/Cat/'):
    try:
        plt.imread(img_dir + '/Cat/' + i)
    except:
        error.append(img_dir + '/Cat/' + i)
        print(img_dir + '/Cat/' + i)

print(error)
