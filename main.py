import keras
import json
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam


base_model = keras.applications.MobileNet(weights='imagenet',include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024,activation = 'relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x = Dense(1024,activation = 'relu')(x) #dense layer 2
x = Dense(512,activation = 'sigmoid')(x) #dense layer 3
preds = Dense(8,activation = 'softmax')(x) #final layer with softmax activation

model = Model(inputs=base_model.input,outputs = preds)

for layer in model.layers[:20]:
    layer.trainable=False
for layer in model.layers[20:]:
    layer.trainable=True

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# train_datagen = ImageDataGenerator(
#                     rotation_range=40,
#                     width_shift_range=0.2,
#                     height_shift_range=0.2,
#                     rescale=1./255,
#                     shear_range=0.2,
#                     zoom_range=0.2,
#                     horizontal_flip=True,
#                     fill_mode='nearest')

train_generator = train_datagen.flow_from_directory('/home/mikes/Pictures/MURA-v1.1/train',
                                                 target_size = (64,64),
                                                 color_mode = 'rgb',
                                                 batch_size = 32,
                                                 #    CHANGE
                                                 class_mode = 'categorical',
                                                 shuffle = True)

# model.load_weights('mod_wChest_GloAVGP_mobile.h5')

# RMSprop
# SGD stohastic gradient descent
adam = Adam(1e-4)# Adam optimizer
model.compile(optimizer='RMSprop',loss='categorical_hinge',metrics=['accuracy'])
                #    CHANGE#    CHANGE
# MSE


step_size_train = train_generator.n // train_generator.batch_size
hist = model.fit_generator(generator = train_generator,
                   steps_per_epoch = step_size_train,
                   epochs = 2)

with open('mod_wChest_RMS_hinge_mobile.json', 'w') as f:
    json.dump(hist.history, f)

model.save('mod_wChest_RMS_hinge_mobile.h5')