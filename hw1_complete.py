#!/usr/bin/env python
# coding: utf-8


# TensorFlow and tf.keras
import tensorflow as tf
import keras
from keras import Input, layers, Sequential

# Helper libraries
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
import json
import sys

print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")


input_shape = (32,32,3)
EPOCHS = 2


def build_model1():
  model = tf.keras.Sequential([
    Input(shape=input_shape),
    layers.Flatten(),
    layers.Dense(128, activation='leaky_relu'),
    layers.Dense(128, activation='leaky_relu'),
    layers.Dense(128, activation='leaky_relu'),    
    layers.Dense(10)
  ])
  return model

def build_model2():
  model = tf.keras.Sequential([
    Input(shape=input_shape),
    layers.Conv2D(32, kernel_size=(3,3), strides=(2,2), 
                  activation="relu", padding='same'),
    layers.BatchNormalization(),
    
    layers.Conv2D(64, kernel_size=(3,3), strides=(2,2),
                  activation="relu", padding='same'),
    layers.BatchNormalization(), 
    
    layers.Conv2D(128, kernel_size=(3,3), activation="relu", padding='same'),
    layers.BatchNormalization(), 
    
    layers.Conv2D(128, kernel_size=(3,3), activation="relu", padding='same'),
    layers.BatchNormalization(), 
    
    layers.Conv2D(128, kernel_size=(3,3), activation="relu", padding='same'),
    layers.BatchNormalization(), 
    
    layers.Conv2D(128, kernel_size=(3,3), activation="relu", padding='same'),
    layers.BatchNormalization(), 
    
    layers.Flatten(),
    layers.Dense(10)
  ])
  return model

def build_model3():
  model = tf.keras.Sequential([
    Input(shape=input_shape),
    layers.SeparableConv2D(32, kernel_size=(3,3), strides=(2,2), 
                  activation="relu", padding='same'),
    layers.BatchNormalization(),
    
    layers.SeparableConv2D(64, kernel_size=(3,3), strides=(2,2),
                  activation="relu", padding='same'),
    layers.BatchNormalization(), 
    
    layers.SeparableConv2D(128, kernel_size=(3,3), activation="relu", padding='same'),
    layers.BatchNormalization(), 
    
    layers.SeparableConv2D(128, kernel_size=(3,3), activation="relu", padding='same'),
    layers.BatchNormalization(), 
    
    layers.SeparableConv2D(128, kernel_size=(3,3), activation="relu", padding='same'),
    layers.BatchNormalization(), 
    
    layers.SeparableConv2D(128, kernel_size=(3,3), activation="relu", padding='same'),
    layers.BatchNormalization(), 
    
    layers.Flatten(),
    layers.Dense(10)
  ])
  return model
  
def build_model50k():
  model = tf.keras.Sequential([
    Input(shape=input_shape),
    layers.SeparableConv2D(32, kernel_size=(3,3), strides=(2,2), 
                  activation="relu", padding='same'),
    layers.BatchNormalization(),
    
    layers.SeparableConv2D(64, kernel_size=(3,3), strides=(2,2),
                  activation="relu", padding='same'),
    layers.BatchNormalization(), 
    
    layers.SeparableConv2D(128, kernel_size=(3,3), activation="relu", padding='same'),
    layers.BatchNormalization(), 
    
    layers.SeparableConv2D(128, kernel_size=(3,3), activation="relu", padding='same'),
    layers.BatchNormalization(), 
    layers.GlobalMaxPool2D(),
    layers.Flatten(),
    layers.Dense(10)
  ])
  return model


def build_model50k_old():
  inputs = Input(shape=input_shape)
  x = layers.Conv2D(32, kernel_size=(3,3), strides=(2,2), 
                    activation="relu", padding='same')(inputs)
  y = layers.BatchNormalization()(x) # save the result in y for the skip connection
  
  
  x = layers.SeparableConv2D(64, kernel_size=(3,3), strides=(2,2), 
                             activation="relu", padding='same')(y)
  x = layers.BatchNormalization()(x)
  
  x = layers.SeparableConv2D(64, kernel_size=(3,3), strides=(2,2),
                             activation="relu", padding='same')(x)
  x = layers.BatchNormalization()(x)
  
  # match the channel counts for the skip ADD.
  y = layers.Conv2D(64, kernel_size=(1,1), strides=(4,4))(y) 
  y = layers.add((x,y))
  
  x = layers.SeparableConv2D(64, kernel_size=(3,3), activation="relu", padding='same')(y)
  x = layers.BatchNormalization()(x)
  
  x = layers.SeparableConv2D(64, kernel_size=(3,3), activation="relu", padding='same')(x)
  x = layers.BatchNormalization()(x)
  
  y = layers.add((x,y))
  
  x = layers.SeparableConv2D(64, kernel_size=(3,3), activation="relu", padding='same')(x)
  x = layers.BatchNormalization()(x)
  
  x = layers.SeparableConv2D(64, kernel_size=(3,3), activation="relu", padding='same')(x)
  x = layers.BatchNormalization()(x)
  
  y = layers.add((x,y))
  
  x = layers.MaxPooling2D(pool_size=(4,4))(x)
  x = layers.Flatten()(x)
  
  x = layers.Dense(64, activation='relu')(x)
  x = layers.BatchNormalization()(x)
  outputs = layers.Dense(10, activation='relu')(x)
  
  model = keras.Model(inputs, outputs)
  return model


def plot_train_history(history):
  plt.subplot(2,1,1)
  plt.plot(history.epoch, history.history['accuracy'], '.-', label="Accuracy")
  if "val_accuracy" in history.history:
    plt.plot(history.epoch, history.history['val_accuracy'], '.-', label="Val Accuracy")
  plt.legend()
  plt.subplot(2,1,2)
  plt.plot(history.epoch, history.history['loss'], '.-')
  if "val_loss" in history.history:
    plt.plot(history.epoch, history.history['val_loss'], '.-', label="Val Loss")
  plt.legend(['Loss'])

def build_parser():
  arg_parser = argparse.ArgumentParser(description="Training script")
  
  arg_parser.add_argument(
    "--train_model1",
    action="store_true",
    help="Train model 1"
  )
  arg_parser.add_argument(
    "--train_model2",
    action="store_true",
    help="Train model 2"
  )
  arg_parser.add_argument(
    "--train_model3",
    action="store_true",
    help="Train model 3"
  )
  arg_parser.add_argument(
    "--test_model1",
    action="store_true",
    help="Test model 1"
  )
  arg_parser.add_argument(
    "--train_model50k",
    action="store_true",
    help="Train 50k-parameter model"
  )
  arg_parser.add_argument(
    "--test_model50k",
    action="store_true",
    help="Test 50k-parameter model"
  )

  arg_parser.add_argument(
    "--epochs",
    type=int, default=5,
    help="Epochs to train"
  )

  return arg_parser


def get_model_info(model):
  # Create the list of layer info
  layer_types = []
  param_counts = []
  for layer in model.layers:
    layer_types.append(layer.__class__.__name__)
    param_counts.append(layer.count_params())
    
  return layer_types, param_counts


if __name__ == '__main__':

  parser = build_parser()
  args = parser.parse_args()
  
  
  ########################################
  ## Load the CIFAR10 data set
  (train_images, train_labels), (test_images, test_labels) = \
    tf.keras.datasets.cifar10.load_data()
  class_names = ['airplane','automobile','bird','cat','deer',
                 'dog','frog','horse','ship','truck']
  
  # Now separate out a validation set.
  val_frac = 0.1
  num_val_samples = int(len(train_images)*val_frac)
  # choose num_val_samples indices up to the size of train_images, !replace => no repeats
  val_idxs = np.random.choice(np.arange(len(train_images)), size=num_val_samples, replace=False)
  trn_idxs = np.setdiff1d(np.arange(len(train_images)), val_idxs)
  val_images = train_images[val_idxs, :,:,:]
  train_images = train_images[trn_idxs, :,:,:]
  
  val_labels = train_labels[val_idxs]
  train_labels = train_labels[trn_idxs]
  
  train_labels = train_labels.squeeze()
  test_labels = test_labels.squeeze()
  val_labels = val_labels.squeeze()
  
  input_shape  = train_images.shape[1:]
  train_images = train_images / 255.0
  test_images  = test_images  / 255.0
  val_images   = val_images   / 255.0
  print("Training Images range from {:2.5f} to {:2.5f}".format(np.min(train_images), np.max(train_images)))
  print("Test     Images range from {:2.5f} to {:2.5f}".format(np.min(test_images), np.max(test_images)))
  
  ########################################
  
  if args.train_model1:
    print("Training model 1")
    
    ## Build and train model 1
    model1 = build_model1()
    model1.compile(optimizer='adam',
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])
    model1.summary()
    
    train_hist1 = model1.fit(train_images, train_labels, 
                             validation_data=(val_images, val_labels),epochs=args.epochs)
    plot_train_history(train_hist1)
    plt.show()
    model1.save("model1.h5")

  if args.test_model1:
    # Test model 1 on a loaded image
    true_label = 'truck'
    truck_img = np.array(tf.keras.utils.load_img(
      './test_image_truck.png',
      grayscale=False,
      color_mode='rgb',
      target_size=(32,32))
                         )
    
    truck_img = np.expand_dims(truck_img, 0)
    plt.imshow(truck_img[0])

    model1 = keras.models.load_model("./model1.h5")

    y = model1.predict(truck_img)
    label_idx = np.argmax(y)
    predicted_label = class_names[label_idx]
    print(f"Image is a {true_label}.  It was labeled as {predicted_label}.")
    print(f"So it was {'CORRECT' if true_label == predicted_label else 'WRONG'}")
  
  
  if args.train_model2:
    print("Training model 2 -- standard convolutions")
    model2 = build_model2()
    model2.compile(optimizer='adam',
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])
    model2.summary()
    
    train_hist2 = model2.fit(train_images, train_labels, 
                          validation_data=(val_images, val_labels),epochs=args.epochs)
    plot_train_history(train_hist2)
    plt.show()

  if args.train_model3:
    print("Training model 3 -- DS Convolutions")
    model3 = build_model3()
    
    model3.compile(optimizer='adam',
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])
    model3.summary()
  
    train_hist3 = model3.fit(train_images, train_labels, 
                          validation_data=(val_images, val_labels),epochs=args.epochs)
    plot_train_history(train_hist3)
    plt.show()

    
  if args.train_model50k:
    print("Training 50k model")

    ## Build and train 50k-param model
    model50k = build_model50k()
    model50k.compile(optimizer='adam',
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])
    
    model50k.summary()
    
    train_hist_50k = model50k.fit(train_images, train_labels,
                                  validation_data=(val_images, val_labels),
                                  epochs=args.epochs)
    plot_train_history(train_hist_50k)
    plt.show()
    model50k.save("best_model.h5")
  

  if args.test_model50k:
    print("Testing 50k model")
    model50k = keras.models.load_model("./best_model.h5")
    model50k.evaluate(val_images, val_labels)

