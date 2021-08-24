import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers.experimental.preprocessing import CenterCrop

from tensorflow.keras import layers,models



#None means the size is unknown
inputs = layers.Input(shape=(None,None, 3))

#adding some preprocessing
cropper =CenterCrop(height=40, width=40)(inputs)
rescaler =Rescaling(scale=1.0 / 255)(cropper)



#adding the convolutions to pick up some features.
#output_shape = ceil(float(ipt_size-filter_size + 1)/float(stride[1]))
x = layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))(rescaler)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)


#adding the classifier at the end, -> they are a dense NN
x = layers.Flatten()(x)
x = layers.Dense(64, activation='relu')(x)
#output layer is a dense layer with 2 outputs as this is the number of classes
#model.add(layers.Dense(2, activation='softmax'))
outputs = layers.Dense(1)(x)

#building the model after the functional definition
model = keras.Model(inputs=inputs, outputs=outputs)



#compiling the model, Binary CE is used as we are predicting two classes
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])



# training dataset. 
dataset_images_training = keras.preprocessing.image_dataset_from_directory(
  '../datasets/teams/train/', 
  batch_size=4, 
  image_size=(40, 50),
  labels='inferred',
  subset="training",
  validation_split=0.2,
  seed=123)


# validation  dataset. 
dataset_images_validation = keras.preprocessing.image_dataset_from_directory(
  '../datasets/teams/train/', 
  validation_split=0.2,
  subset="validation",
  seed=123,
  batch_size=4,
  image_size=(40, 50),
  labels='inferred')


model.summary()


callbacks_custom = [
    keras.callbacks.ModelCheckpoint(
        filepath='functional_saved/classification_model_epoch_{epoch}',
        save_freq='epoch'
        ),
        keras.callbacks.TensorBoard(log_dir='./functional_logs')
]

history=model.fit(
  dataset_images_training,
  validation_data=dataset_images_validation,
  epochs=2,
  callbacks=callbacks_custom
)

print(history.history)

test_loss, test_acc = model.evaluate(dataset_images_validation, verbose=2)
print('\n',test_loss,test_acc)

predictions=model.predict(dataset_images_validation)

for pred in predictions:
    print(pred)
    print('done')