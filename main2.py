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
# x = Dense(1024,activation = 'relu')(x) #dense layer 2
x = Dense(512,activation = 'relu')(x) #dense layer 3
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
                                                 target_size = (32,32),
                                                 color_mode = 'rgb',
                                                 batch_size = 32,
                                                 class_mode = 'categorical',
                                                 shuffle = True)

# model.load_weights('mod_wChest_4lay_VGG19.h5')

adam = Adam(1e-3)# Adam optimizer
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])


step_size_train = train_generator.n // train_generator.batch_size
history = model.fit_generator(generator = train_generator,
                   steps_per_epoch = step_size_train,
                   epochs = 3)

valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
valid_generator = valid_datagen.flow_from_directory('/home/mikes/Pictures/MURA-v1.1/valid',
                                                 target_size = (32,32), #target_size: tuple of integers (height, width), default: (256, 256). The dimensions to which all images found will be resized.
                                                 color_mode = 'rgb',
                                                 batch_size = 3,
                                                 class_mode = 'categorical')

step_size_valid = valid_generator.n // valid_generator.batch_size

print("The Validation: ")
print(model.predict_generator(generator = valid_generator,
                             steps = step_size_valid))

with open('history_wChest_4lay_VGG19.json', 'w') as f:
    json.dump(history.history, f)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# model.save('mod_wChest_4lay_VGG19.h5')
