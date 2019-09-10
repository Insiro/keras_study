import keras
from keras import layers, Input, optimizers, models
from keras.applications.resnet50 import ResNet50, decode_predictions
import datetime as dt
import numpy as np
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input
img_size = (224, 224)
BATCH_SIZE = 1
NUM_EPOCHS = 100
DATAPATCH = '../2019_endoscopy_image/'
classes = ['normal', 'abnormal']

# generate dataset
train_datagen = ImageDataGenerator(preprocess_input,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   fill_mode='nearest')
train_batches = train_datagen.flow_from_directory(
    directory=DATAPATCH+'train', target_size=img_size, batch_size=BATCH_SIZE, class_mode='binary', classes=classes)
test_datagen = ImageDataGenerator(preprocess_input)
valid_batches = test_datagen.flow_from_directory(
    directory=DATAPATCH+'test', target_size=img_size, batch_size=BATCH_SIZE, class_mode='binary', classes=classes)
input_tensor = Input(shape=(124, 124, 3), dtype='float32', name='input')

# load resnet 50
base_model = ResNet50(weights='imagenet', include_top=False,
                      input_shape=(224, 224, 3))
print(base_model.summary())

# tensor board set
logdir = "logs/"+dt.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

# define classifier
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(2048, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
model1 = Model(inputs=base_model.input, outputs=predictions)
model1.compile(optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
               loss='binary_crossentropy', metrics=['accuracy'])

# reduce LR when stuff
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=8, min_lr=0.0000001, verbose=1, cooldown=1)

# fit
H = model1.fit_generator(train_batches,
                         steps_per_epoch=train_batches.samples // BATCH_SIZE,
                         validation_data=valid_batches,
                         validation_steps=valid_batches.samples // BATCH_SIZE,
                         epochs=NUM_EPOCHS,
                         callbacks=[tensorboard_callback, reduce_lr]).history

# printout resualt
history = H.history
loss = history['loss']
val_loss = history['val_loss']
acc = history['acc']
val_acc = history['val_acc']
epochs = range(1, len(loss)+1)
epochs2 = range(1, len(val_loss)+1)
plt.plot(epochs, loss, 'r', label='train loss')
plt.plot(epochs, acc, 'g', label='train acc')
plt.plot(epochs2, val_acc, 'b', label='Val acc')
plt.plot(epochs2, val_loss, 'c', label='Val loss')
plt.xlabel('Epochs')
plt.ylabel('Loss/Acc')
plt.legend()
plt.show()
plt.plot(epochs, loss, 'r', label='train loss')
plt.plot(epochs2, val_loss, 'c', label='Val loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.plot(epochs, acc, 'g', label='train acc')
plt.plot(epochs2, val_acc, 'b', label='Val acc')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.show()
# result
test_batches = train_datagen.flow_from_directory(DATAPATCH + 'test',
                                                 target_size=img_size,
                                                 class_mode='binary',
                                                 batch_size=BATCH_SIZE,
                                                 classes=classes,
                                                 shuffle=False)
result = model1.evaluate_generator(
    generator=test_batches, steps=test_batches.samples)
print(result)
