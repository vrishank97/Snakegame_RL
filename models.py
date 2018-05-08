import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

def LightCNN():
	model = Sequential()

	model = Sequential()
	model.add(Conv2D(32, (5, 5), input_shape=(1, 10, 15), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(50, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))

	model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

	return model