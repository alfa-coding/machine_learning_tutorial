
import numpy as np
import tensorflow as tf
from tensorflow import keras

#we dont know the input sizes
IMG_SHAPE= None

base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

base_model.summary()

#Freezing the base so we dont train the already learned knowledge 
#encoded into its millions of parameters. This is called Transfer learning
base_model.trainable = False

#creates the 1st layer which basically instead of flattening will scale down the day pooling it to its average, check pooling above
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()



#now we crate a dense layer to which all the neurons of the previous layer created connect to, in
#this case it has one element cuz we want to predict whether it is of class A or B
prediction_layer = keras.layers.Dense(1)

model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])

#Check that we are only training the classifier at the end of the model
#the rest is frozen
model.summary()


#compiling the model, Binary CE is used as we are predicting two classes
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
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




callbacks_custom = [
    keras.callbacks.ModelCheckpoint(
        filepath='usingBaseModel_saved/classification_model_epoch_{epoch}',
        save_freq='epoch'
        ),
        keras.callbacks.TensorBoard(log_dir='./usingBaseModel_logs')
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

