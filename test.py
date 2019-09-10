import keras
from keras import layers, Input, optimizers, models
from keras.applications.vgg16 import VGG16
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
img_size = (224, 224)
BATCH_SIZE = 2
NUM_EPOCHS = 5
DATAPATCH = '2019_endoscopy_image/'

classes = ['normal', 'abnormal']
train_datagen = ImageDataGenerator(rescale=1./255)
train_data = train_datagen.flow_from_directory(
    directory=DATAPATCH+'train', target_size=img_size, batch_size=BATCH_SIZE, class_mode='binary', classes=classes)
test_datagen = ImageDataGenerator(rescale=1./255)
test_data = test_datagen.flow_from_directory(
    directory=DATAPATCH+'test', target_size=img_size, batch_size=BATCH_SIZE, class_mode='binary', classes=classes)
input_tensor = Input(shape=(224, 224, 3), dtype='float32', name='input')
pre_VGG = VGG16(weights='imagenet', include_top=False,
                input_shape=(224, 224, 3))
pre_VGG.trainable = False
pre_VGG.summary()


allModel = models.Sequential()
allModel.add(pre_VGG)
allModel.add(layers.Flatten())
allModel.add(layers.Dense(512, activation='relu'))

allModel.add(layers.Dense(1, activation='sigmoid'))
allModel.compile(optimizer='rmsprop',
                 loss='binary_crossentropy', metrics=['accuracy'])

allModel.summary()
# history = allModel.fit_generator(train_data, epochs=NUM_EPOCHS, validation_steps=test_data.samples,
#                                  steps_per_epoch=train_data.samples, validation_data=test_data).history
# loss = history['loss']
# val_loss = history['val_loss']
# acc = history['acc']
# val_acc = history['val_acc']
# epochs = range(1, len(loss)+1)
# plt.plot(epochs, loss, 'r', label='training loss')
# plt.plot(epochs, val_loss, 'c', lable='Validation loss')
# plt.plot(epochs, acc, 'g', label='training acc')
# plt.plot(epochs, val_acc, 'b', lable='Validation acc')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
