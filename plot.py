import keras
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam

img_data = []

with open('/home/mikes/Downloads/ctlos.tsv') as f:
    read_data1 = f.read().split("\n")
    read_data1 = [float(y) for y in read_data1]
with open('/home/mikes/Downloads/ctacc.tsv') as f:
    read_data2 = f.read().split("\n")
    read_data2 = [float(y) for y in read_data2]
# with open('/home/mikes/Downloads/rsm1.tsv') as f:
#     read_data3 = f.read().split("\n")
#     read_data3 = [float(y)*100 for y in read_data3]


x = range(len(read_data1))
plt.plot(x,read_data1, 'r', label='Loss')
plt.plot(x,read_data2, 'b', label='Accuracy')
# plt.plot(x,read_data3, 'b', label='RMSprop')
plt.title('Model accuracy and loss in one epoch')
plt.ylabel('')
plt.xlabel('#batches')
# plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.0f%%'))
plt.legend(loc='lower middle')
plt.show()

plt.savefig('')

# with open('/home/mikes/Downloads/b1.tsv') as f:
#     read_data = f.read().split("\n")
#     read_data = [float(y) for y in read_data]
#
# x = range(len(read_data))
# plt.plot(x,read_data)
# plt.show()