from keras.models import Sequential
from keras.models import Dense, Activation, Flatten

# Sequential model
model  = Sequential()

# Layers
## Layer 1 - Add a flatten layer
model.add(Flatten(input_shape=(32, 32, 3)))

## Layer 2 - add a fully connected layer
model.add(Dense(100))

## Layer 3 - add a ReLU activation layer
model.add(Activation('relu'))

## Layer 4 -  add a fully connected layer
model.add(Dense(60))

## Layer 5 - add a ReLU activation layer
model.add(Activation('relu'))
