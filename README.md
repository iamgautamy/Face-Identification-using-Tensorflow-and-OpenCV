# Face-Identification-using-Tensorflow-and-OpenCV
This is a TensorFlow implementation for face identification from scratch using CNN layers.

This was made in Tensorflow Version 2.8.0, Python 3.8 under Anaconda Virtual Environment

The model is trained on Several Images from the dataset and with my own face added

# Layers Used:
  tf.keras.layers.Rescaling(1./255),
  
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  
  tf.keras.layers.MaxPooling2D(),
  
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  
  tf.keras.layers.MaxPooling2D(),
  
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  
  tf.keras.layers.MaxPooling2D(),
  
  tf.keras.layers.Flatten(),
  
  tf.keras.layers.Dense(128, activation='relu'),
  
  tf.keras.layers.Dense(num_classes)

# Accuracy and Confidence of Model
Model shows an approx of 84% accuracy on both training and validation dataset
And has mean confidence on faces over 90%





