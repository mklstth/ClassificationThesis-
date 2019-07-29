import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import numpy as np
import json
from keras.optimizers import Adam


base_model = keras.applications.mobilenet.MobileNet(weights='imagenet',include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024,activation = 'relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x = Dense(1024,activation = 'relu')(x) #dense layer 2
x = Dense(512,activation = 'sigmoid')(x) #dense layer 3
preds = Dense(8,activation = 'softmax')(x) #final layer with softmax activation

model = Model(inputs=base_model.input,outputs = preds)
model.load_weights('mod_wChest_SGD_CC_mobile.h5')

matrix = []

for x in range(1, 9):
    valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    valid_generator = valid_datagen.flow_from_directory('/home/mikes/Pictures/MURA-v1.1/cross/' + str(x),
                                                     target_size = (64,64), #target_size: tuple of integers (height, width), default: (256, 256). The dimensions to which all images found will be resized.
                                                     color_mode = 'rgb',
                                                     batch_size = 1,
                                                     class_mode = 'categorical')

    step_size_valid = valid_generator.n // valid_generator.batch_size

    valid = model.predict_generator(generator = valid_generator,
                                 steps = step_size_valid)


    valid = np.array(valid)
    valid2 = []

    for row in valid:
        row2 = np.zeros(8)
        loc = np.argmax(row)
        row2[loc] = 1
        valid2.append(row2)

    valid = np.array(valid2)

    valid = np.sum(valid, axis = 0) / step_size_valid
    print(valid)

    matrix.append(valid)

# print(matrix)
# with open('matrix.txt', 'w') as file:
#     json.dump(matrix.data, file)

plt.imshow(matrix, interpolation='nearest')
plt.title('Confusion matrix')
plt.yticks(np.arange(8),('Chest', 'Elbow', 'Finger', 'Forearm', 'Hand', "Humerus", 'Shoulder', 'Wrist'))
plt.xticks(np.arange(8),('Chest', 'Elbow', 'Finger', 'Forearm', 'Hand', "Humerus", 'Shoulder', 'Wrist'), rotation=45)
for i in range(len(matrix)):
    for j in range(len(matrix)):
        text = plt.text(j, i, round(matrix[i][j],2),
                       ha="center", va="center", color="m")


plt.gcf()
plt.savefig('heatmapMobile.png')
plt.show()



# plt.xticks(range(len(t11)), t11)
