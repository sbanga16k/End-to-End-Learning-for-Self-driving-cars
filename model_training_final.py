import numpy as np
import cv2
import tensorflow as tf

import model_final
import augment_helper_fns

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from csv import reader

# from keras import backend
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

flags = tf.app.flags
FLAGS = flags.FLAGS

# Command line flags
flags.DEFINE_integer('dataset_idx', 3,"Dataset index:")
flags.DEFINE_integer('epochs', 10, "The number of epochs:")
flags.DEFINE_integer('batch_size', 64, "The batch size:")

use_generator = False		# Flag indicating whether to use generator or not. Set to FALSE since Generator not utilized for training
idx = 11					# Index for file names acc. to models


def read_csv(file_name, obs_path):
	"""
	Reads in rows of specified csv file and stores it in a list
	"""
	with open(file_name) as f:
		csv_reader = reader(f)
		for row in csv_reader:
			if file_name.split('/')[-2] == "Track1_Udacity":
				locs = []
				for elem in row[0:3]:
					locs.append(elem.replace('/','\\'))
				locs.extend(row[3:])
				obs_path.append(locs)
			else:
				obs_path.append(row)
	return obs_path

def load_csv():
	"""
	Returns list with paths of images & other driving attribute values (steering, throttle) for the 
	specified dataset along with name of directory having images
	"""

	print("\nLoading CSV data...")

	if FLAGS.dataset_idx == 1:
		fname = "./data/Track1_Udacity/driving_log.csv"
	elif FLAGS.dataset_idx == 2:
		fname = "./data/Track1_custom/driving_log.csv"
	elif FLAGS.dataset_idx == 3:
		fname = "./data/Customdata_jungle/driving_log.csv"
	else:
		raise ValueError("Index out of bounds!")

	### Reading data from csv and storing it in a list by rows
	samples_path = []

	samples_path = read_csv(fname, samples_path)

	print("Loading complete!")

	return samples_path, fname


def preprocessing(img):
	"""
	Applying all required pre-processing actions (Cropping, resizing & color scale conversion)
	"""

	# Crop the top and bottom portions of the image to remove the sky, irrelevant surroundings
	# and also the hood of the car - all of which might confuse the model

	# Margin for trimming from top nd bottom of image
	(crop_top,crop_bot) = (40,20)
	img = img[crop_top:(img.shape[0]-crop_bot), :, :]
	
	# Alt 1a: Resize the image to size that the NVIDIA model utilizes
	# img = cv2.resize(img, (200, 66), cv2.INTER_AREA)

	# Alt 1b: Resize the image to smaller size for ease of storage & computation without impacting performance
	img = cv2.resize(img, (64, 64), cv2.INTER_AREA)

	# Alt 3: Convert the image from RGB to YUV (This is what the NVIDIA model does)
	img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

	return img


def data_augument(img, steer_angle, range_x=60, range_y=15):
	"""
	Generate an augumented image and its corresponding steering angle
	"""
	if (np.random.random() <= 0.5):
		img, steer_angle = augment_helper_fns.flip_img(img, steer_angle)
	if (np.random.random() <= 0.4):
		img, steer_angle = augment_helper_fns.transl_img(img, steer_angle, range_x, range_y)
	if (np.random.random() >= 0.6):
		img = augment_helper_fns.brightness_aug(img)
	elif (np.random.random() >= 0.7):
		img = augment_helper_fns.shadow_aug(img)
	
	return img, steer_angle


def load_images(sampled_obs, dir_name, train=True):
	"""
	Generates the dataset for training & validation with the necessary pre-processing 
	(cropping, resizing, transforming color spaces) and also data augmentation (during training only) 
	depending on the training flag
	"""
	if train == True:
		print("\nLoading training samples...")
	else:
		print("\nLoading validation samples...")

	images = []
	steer_angles = []

	for row in sampled_obs:
		i = np.random.randint(3)
		og_path = row[i]
		img_name = og_path.split('\\')[-1]
		new_path = "./data/" + dir_name.split('/')[-2] + "/IMG/" + img_name
		
		img = mpimg.imread(new_path)	# Reads RGB image
		steer_angle = float(row[3])

		# Steering angle appears more shallow for images when viewed from left camera
		if i == 1:
			steer_angle += 0.20
		# Steering angle appears wider for images when viewed from right camera
		elif i == 2:
			steer_angle -= 0.20

		# Data augmentation and Pre-processing images
		if train == True:
			img, steer_angle = data_augument(img, steer_angle, range_x=80, range_y=10)
		img = preprocessing(img)

		images.append(img)
		steer_angles.append(steer_angle)

	X_data = np.array(images)
	y_data = np.array(steer_angles)

	print("Dataset created!")

	return shuffle(X_data, y_data)


def data_generator(sampled_obs, batch_size, dir_name, train=True):
	"""
	Loads images using generators to yield data in bacthes after carrying out the appropriate
	pre-processing and data augmentation (augmentation only during training)

	"""

	n_obs = len(sampled_obs)
	# Loop forever so the generator never terminates
	while True:
		shuffle(sampled_obs)
		for start_idx in range(0, n_obs, batch_size):
			batch_obs = sampled_obs[start_idx:start_idx + batch_size]

			images = []
			steer_angles = []

			for row in batch_obs:
				i = np.random.randint(3)
				og_path = row[i]
				img_name = og_path.split('\\')[-1]
				new_path = "./data/" + dir_name.split('/')[-2] + "/IMG/" + img_name
				
				img = mpimg.imread(new_path)	# Reads RGB image
				steer_angle = float(row[3])

				# Steering angle appears more shallow for images when viewed from left camera
				if i == 1:
					steer_angle += 0.20
				# Steering angle appears wider for images when viewed from right camera
				elif i == 2:
					steer_angle -= 0.20

				# Data augmentation and Pre-processing images
				if train == True:
					img, steer_angle = data_augument(img, steer_angle, range_x=80, range_y=10)
				img = preprocessing(img)

				images.append(img)
				steer_angles.append(steer_angle)

			# Converting lists into arrays (format accepted by "Keras")
			X_data = np.array(images)
			y_data = np.array(steer_angles)

			yield shuffle(X_data, y_data)


def main(_):
	samples_path, dir_name = load_csv()
	train_obs_path, valid_obs_path = train_test_split(samples_path, test_size=0.20)

	X_train, y_train = load_images(train_obs_path, dir_name, train=True)
	X_val, y_val = load_images(valid_obs_path, dir_name, train=False)

	og_img_shape = X_train.shape[1:]

	# Checking training and validation data
	assert len(X_train) == len(y_train), "X_train {} & y_train {} are not equal".format(len(X_train),len(y_train))
	assert len(X_val) == len(y_val), "X_val {} & y_val {} are not equal".format(len(X_val),len(y_val))
	print("Number of training samples: {} \nNumber of validation samples:  {}".format(X_train.shape[0], X_val.shape[0]))

	# Compile and train the model using the generator function
	if use_generator:
		train_gen = data_generator(train_obs_path, FLAGS.batch_size, dir_name, train=True)
		valid_gen = data_generator(valid_obs_path, FLAGS.batch_size, dir_name, train=False)
		print("Generators created!")

	### Actual training of the model and saving paameters
	print("Training begins!")

	model = model_final.model_def(og_img_shape)
	model.compile(optimizer = Adam(lr=1e-4), loss='mse')
	checkpointer = ModelCheckpoint('./model-{epoch:02d}-{val_loss:.3f}.h5', monitor='val_loss', verbose=0, \
		save_best_only=True, mode='auto')

	if use_generator:
		history_obj = model.fit_generator(train_gen, steps_per_epoch= len(train_obs_path), validation_data = valid_gen, \
			validation_steps=len(valid_obs_path), epochs=FLAGS.epochs, callbacks = [checkpointer], verbose=1)
	else:
		history_obj = model.fit(X_train, y_train, batch_size=FLAGS.batch_size, epochs=FLAGS.epochs, validation_data = (X_val, y_val), \
			callbacks = [checkpointer], shuffle=True, verbose=1)

	import pickle

	with open("./train_mse_loss"+str(idx)+".p", 'wb') as f:
		pickle.dump({"data": history_obj.history['loss']}, f)
	with open("./val_mse_loss"+str(idx)+".p", 'wb') as f:
		pickle.dump({"data": history_obj.history['val_loss']}, f)

	model.save('./model'+str(idx)+'.h5')

	# Print the keys contained in the history object
	# print(history_obj.history.keys())

	# Plot metrics
	plt.plot(history_obj.history['loss'])
	plt.plot(history_obj.history['val_loss'])
	plt.title('Loss')
	plt.ylabel('Mean squared error')
	plt.xlabel('Epochs')
	plt.legend(['Training loss', 'Validation loss'], loc='upper right')
	plt.show()

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()