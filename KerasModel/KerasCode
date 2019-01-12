#_________________ Keras Model _____________________
import io
import cv2
import pandas as pd
import numpy as np
import base64
import matplotlib.pyplot as plt
import itertools
from PIL import Image
from google.cloud import bigquery
from sklearn.model_selection import train_test_split

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

np.set_printoptions(threshold=np.nan)

col_names = ['VehicleType', 'WeightClass', 'Industry', 'Carrier', 'GovVocation', 'PlaceTypeImage', 'PlaceSubTypeImage', 'NumPlaceTypeElements', 'NumPlaceSubTypeElements', 'labels']

def GBQ():

	allImages = '''

	select * FROM `industryclassification.ImageFeatures.VehicleIndustryPlaceImages`

	'''
	
	df = pd.read_gbq(allImages, project_id='industryclassification', dialect='standard')
	
	data = []
	c = 1

	for index, row in df.iterrows():
		r = []
		for i in range(len(col_names)-1):
			r.append(row[col_names[i]])
		if r[4] == None:
			r.append((r[2]))
		else:
			r.append((r[2]) + "-" + (r[4]))
		data.append(r)
		
	np_array = np.array(data)
	df = pd.DataFrame(np_array)
	df.columns = col_names
	df.to_csv("data.csv")

def GetImage(imageBytes, width, height, dpi):
	buf = io.BytesIO() 
	plt.switch_backend('Agg')
	image = io.BytesIO(base64.standard_b64decode(imageBytes))
	plt.figure(figsize=(width, height), dpi=dpi)
	plt.imshow(plt.imread(image))
	plt.axis('off')
	plt.savefig(buf, format='png')
	buffer = buf.getvalue()
	buf.close()
	return buffer 	
	
def main():
	# uncomment if first time running, will extract from bigQuery -> CSV
	# ----------------------
	
	GBQ()

	# ----------------------
	
	exp = 200	# min in last 2 columns
	df = pd.read_csv("data.csv")	
	df = df[df.NumPlaceSubTypeElements > exp]
	df = df[df.NumPlaceTypeElements > exp]

		
	callbacks = [EarlyStopping(monitor='val_loss', patience=2),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
	
	y = df['labels'].values


	X = []
	
	# uncomment to use cpu
	# ----------------------
	
	# import tensorflow as tf
	# from keras import backend as K

	# config = tf.ConfigProto(
        # device_count = {'GPU': 0}
    # )
	# sess = tf.Session(config=config)
	# K.set_session(sess)
	
	# ----------------------
	
	batch_size = 32
	epochs = 30		# early stopping is applied
	labs = set(y)
	num_classes = len(labs)

	
	for i in range(len(y)):
		y[i] = list(labs).index(y[i])
		
		
		
	c = 1
	for img in df['PlaceTypeImage']:
		# Mike's Method
		image = GetImage(img, 0.25, 1.25, 1000) #1250px x 250px --> 1000 dpi
		image = cv2.imdecode(np.fromstring(image, dtype=np.uint8), 1) # <-- 0=Grayscale
		image = image.astype(np.float32)
		image = np.multiply(image, 1.0 / 255.0)
	
		# Percent completion
		if not c%1000:
			print(str(c/df.shape[0]))
			
		height = image.shape[0]
		width = image.shape[1]

		image.resize(height,width,1)
		X.append(image)
		
		c += 1
	print(1.0)	

	X = np.array(X)
	x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state = 123)
	
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)
	
	
	scores = []
	
	kernel_sizes = [(2, 2),(3, 3)] #,(4, 4),(5, 5)]
	pool_sizes = [(2, 2),(3, 3)]
	dropout1 = [0.2,0.25] #,0.3]
	dropout2 = [0.4,0.5] #,0.6]
	optimizers = ['adam',keras.optimizers.Adadelta()] #,keras.optimizers.SGD(lr=0.01, clipnorm=1.)] # adam, adadelta, sgd
	
	gridSearchNetwork = list(itertools.product(kernel_sizes,pool_sizes,dropout1,dropout2,optimizers))

	use = [0,2,12,6,14,20,24] #only used for testing purposes
	
	c = 0
	print("Initiating Model")
	for i in gridSearchNetwork:
		if c in use:
			model = Sequential()
			model.add(Conv2D(32, kernel_size=i[0],
							 activation='relu',
							 input_shape=(height,width,1)))
			model.add(Conv2D(64, i[0], activation='relu'))
			model.add(MaxPooling2D(pool_size=i[1]))
			model.add(Dropout(i[2]))
			model.add(Flatten())
			model.add(Dense(128, activation='relu'))
			model.add(Dropout(i[3]))
			model.add(Dense(num_classes, activation='softmax'))
			
			print("Compiling")
			model.compile(loss=keras.losses.categorical_crossentropy,
						  optimizer=i[4],
						  metrics=['accuracy'])	

			print("fitting")
			model.fit(x_train, y_train,
					  batch_size=batch_size,
					  epochs=epochs,
					  callbacks=callbacks,
					  verbose=1,
					  validation_data=(x_test, y_test))

			print("scoring")
			score = model.evaluate(x_test, y_test, verbose=0)
			print('Test loss:', score[0])
			print('Test accuracy:', score[1])
			print(i)
			if score[1] > 0.45:
				scores.append([score[1],c,i])
		c += 1
		
	print('best score and params')	
	print(scores)
	
main()
