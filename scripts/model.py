import settings
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Dropout
import time
import numpy as np
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


class DigitClassifier:

	@staticmethod
	def build_model():
		# Grid search The best hyper parameters =  {'activation_mode': 'relu', 'init_mode': 'he_normal', 'optimizer': 'adam'}
		model = Sequential([
			Flatten(input_shape=(28, 28), name='input'),
			Dense(100, activation='relu', kernel_initializer='he_normal', name='hidden_1'),
			Dropout(0.2),
			Dense(100, activation='relu', kernel_initializer='he_normal', name='hidden_2'),
			Dropout(0.2),
			Dense(100, activation='relu', kernel_initializer='he_normal', name='hidden_3'),
			Dropout(0.2),
			Dense(100, activation='relu', kernel_initializer='he_normal', name='hidden_4'),
			Dropout(0.2),
			Dense(100, activation='relu', kernel_initializer='he_normal', name='hidden_5'),
			Dropout(0.2),
			Dense(5, activation=tf.nn.softmax, name='output')
		])
		return model

	def train_model(self, model, X_train, y_train):
		start_time = time.time()

		model.compile(optimizer='adam',
					  loss='sparse_categorical_crossentropy',
					  metrics=['accuracy'])

		tensorboard = TensorBoard(log_dir=settings.lOGS_DIR)
		early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
													   min_delta=0.005,
													   patience=2,
													   verbose=0,
													   mode='auto')
		history = model.fit(X_train, y_train,
							validation_split=0.2,
							epochs=20,
							callbacks=[tensorboard, early_stopping])

		end_time = time.time()
		print('Total train time = ', round(end_time - start_time), 's')
		self._visualize_model_training(history)
		return model

	@staticmethod
	def evaluate_model(model, X_test, y_test):
		test_loss, test_acc = model.evaluate(X_test, y_test)
		print('Test accuracy = ', test_acc)

	# It need for grid research
	@staticmethod
	def _create_model(activation_mode='elu', init_mode='he_normal', optimizer='adam'):
		model = keras.models.Sequential([
			tf.keras.layers.Flatten(input_shape=(28, 28)),
			tf.keras.layers.Dense(100, activation=activation_mode, kernel_initializer=init_mode),
			tf.keras.layers.Dense(100, activation=activation_mode, kernel_initializer=init_mode),
			tf.keras.layers.Dense(100, activation=activation_mode, kernel_initializer=init_mode),
			tf.keras.layers.Dense(100, activation=activation_mode, kernel_initializer=init_mode),
			tf.keras.layers.Dense(100, activation=activation_mode, kernel_initializer=init_mode),
			tf.keras.layers.Dense(5, activation=tf.nn.softmax)
		])
		model.compile(optimizer=optimizer,
					  loss='sparse_categorical_crossentropy',
					  metrics=['accuracy'])
		return model

	def get_best_parameters(self, X_train, y_train):
		model = KerasClassifier(build_fn=self._create_model)
		param_grid = [
			{
				'init_mode': ['he_normal', 'he_uniform', 'glorot_normal'],
			 	'activation_mode': ['elu', 'relu'],
				'optimizer': ['adam', 'Nadam']
			}
		]
		grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
								   n_jobs=-1)

		grid_search.fit(X_train, y_train)
		print('The best hyper parameters = ', grid_search.best_params_)

	@staticmethod
	def _visualize_model_training(history):
		print(history.history.keys())
		plt.title('Error model')
		plt.ylabel('loss')
		plt.xlabel('Number of epoch')
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.legend(['train, validation'], loc='upper left')
		plt.show()

	@staticmethod
	def save_model(model, model_type):
		if model_type == 'model 1':
			file_name = settings.MODEL_1_FILE_NAME
		else:
			file_name = settings.MODEL_2_FILE_NAME
		try:
			keras.models.save_model(model, settings.MODELS_DIR + file_name)
		except IOError:
			raise ValueError('Something wrong with file save operation.')
		except ValueError:
			raise ValueError('Something wrong with model.')

	@staticmethod
	def load_my_model(model_type):
		if model_type == 'model 1':
			file_name = settings.MODEL_1_FILE_NAME
		else:
			file_name = settings.MODEL_2_FILE_NAME
		try:
			model = keras.models.load_model(settings.MODELS_DIR + file_name, compile=False)
			return model

		except IOError:
			raise ValueError('Something wrong with file read operation.')

	@staticmethod
	def rebuild_model(parent_model):
		print('Parent model')
		parent_model.summary()

		# get hidden parent layers
		parent_input_layer = parent_model.get_layer('input')
		parent_hidden_layer_1 = parent_model.get_layer('hidden_1')
		parent_hidden_layer_2 = parent_model.get_layer('hidden_2')
		parent_hidden_layer_3 = parent_model.get_layer('hidden_3')
		parent_hidden_layer_4 = parent_model.get_layer('hidden_4')
		parent_hidden_layer_5 = parent_model.get_layer('hidden_5')

		# give all parent model hidden layers
		# freeze 2 lowest layers of them
		# add two new hidden layers
		rebuilt_model = Sequential([
			parent_input_layer,
			parent_hidden_layer_1,
			Dropout(0.2),
			parent_hidden_layer_2,
			Dropout(0.2),
			parent_hidden_layer_3,
			Dropout(0.2),
			parent_hidden_layer_4,
			Dropout(0.2),
			parent_hidden_layer_5,
			Dropout(0.2),
			Dense(100, activation='relu', kernel_initializer='he_normal', name='hidden_6'),
			Dropout(0.2),
			Dense(100, activation='relu', kernel_initializer='he_normal', name='hidden_7'),
			Dropout(0.2),
			Dense(5, activation=tf.nn.softmax, name='output')
		])

		# freeze 2 lowest parent layers
		rebuilt_model.layers[1].trainable = False
		rebuilt_model.layers[3].trainable = False

		rebuilt_model.summary()
		return rebuilt_model

	@staticmethod
	def predict_number(model, x, y):

		x = np.expand_dims(x, axis=0)
		predicted = int(model.predict_classes(x))

		# mapper
		y_real = y + 5
		predicted += 5

		print('Predict =', predicted)
		print('Label = ', y_real)
