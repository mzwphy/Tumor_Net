

import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn import preprocessing


#LOAD IMAGE DATA
################

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    rescale=1/255,
    horizontal_flip=True

)

test_datagen = ImageDataGenerator(
    rescale=1/255,

)

train_generator = train_datagen.flow_from_directory(
    directory="/home/mzwandile/Downloads/TumorNet/data/train/", #put here your path to training images
    target_size=(224, 224), 
    batch_size=32,
    class_mode='categorical', 
    shuffle=True,
    seed=42,
    color_mode="rgb",
)

val_generator = val_datagen.flow_from_directory(
    directory="/home/mzwandile/Downloads/TumorNet/data/valid/", #put here your path to validation images
    target_size=(224,224),
    class_mode='categorical', 
    batch_size=32,
    shuffle=True,
    seed=42,
    color_mode="rgb"
)

test_generator = test_datagen.flow_from_directory(
    directory="/home/mzwandile/Downloads/TumorNet/data/test/", #put here your path to test images
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=1,
    class_mode=None, 
    shuffle=False,
    seed=42
)


#BUILD MODEL
############

kernel_size = (3, 3)
pool_size = (2, 2)

model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, kernel_size = kernel_size, activation="relu", input_shape=(224,224,3)),
    tf.keras.layers.MaxPooling2D(pool_size = pool_size),
    #tf.keras.layers.Dropout(0.5),


    tf.keras.layers.Conv2D(64, kernel_size = kernel_size, activation="relu", padding = 'same'),
    tf.keras.layers.MaxPooling2D(pool_size = pool_size),
    tf.keras.layers.Dropout(0.5),


    tf.keras.layers.Conv2D(64, kernel_size = kernel_size, activation="relu", padding = 'same'),
    tf.keras.layers.MaxPooling2D(pool_size = pool_size),
    tf.keras.layers.Dropout(0.5),


    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(250, activation="relu"),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(3, activation = 'softmax')

  ]
)

#COMPILE MODEL
##############
metriks = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.TruePositives(name="TP"),
    tf.keras.metrics.TrueNegatives(name="TN"),
    tf.keras.metrics.FalsePositives(name="FP"),
    tf.keras.metrics.FalseNegatives(name="FN"),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
    tf.keras.metrics.AUC(name='auc'),
    tf.keras.metrics.AUC(name='prc', curve='PR') # precision-recall curve
]


model.compile(
	optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=metriks
)

model.summary()

#TRAIN MODEL
############
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=val_generator.n//val_generator.batch_size
history = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=val_generator,
    validation_steps=STEP_SIZE_VALID,
    epochs=25,
    verbose=1
)

#SAVE MODEL
###########
model.save('/home/mzwandile/Downloads/TumorNet/run/TumorNet.h5')

#VISUALIZATIONS
###############
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
def plot_metrics(history):
    metrics = ['loss', 'auc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='train')
        plt.plot(history.epoch, history.history['val_'+metric],
             color=colors[0], linestyle="--", label='val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.3,1.0])
        else:
            plt.ylim([0,1])

            plt.legend()
    plt.show()

plot_metrics(history)

#VISUALIZATIONS 
############### 
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss'] 
val_loss = history.history['val_loss']

plt.figure(figsize=(4, 4))
plt.subplot(2, 1, 1)
plt.plot(acc, label='train acc')
plt.plot(val_acc, label='val acc')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')
  
plt.subplot(2, 1, 2)
plt.plot(loss, label='train loss')
plt.plot(val_loss, label='val loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


#PREDICT OUTPUT
###############

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
test_generator.reset()
pred=model.predict_generator(
    test_generator,
    steps=STEP_SIZE_TEST,
    verbose=1
)

predicted_class_indices=np.argmax(pred,axis=1)
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

filenames=test_generator.filenames
results=pd.DataFrame({
    "Filename":filenames,
    "Predictions":predictions
})
results.to_csv("results.csv",index=False)

print(results.head(50))
