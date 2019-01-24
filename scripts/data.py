import tensorflow as tf
import matplotlib.pyplot as plt
mnist = tf.keras.datasets.mnist


class MNISTData:

	def get_train_and_test_data_0_4(self):
		X_train_full, y_train_full, X_test_full, y_test_full = self.get_full_train_and_test_data()
		X_train_0_4 = X_train_full[y_train_full < 5]
		y_train_0_4 = y_train_full[y_train_full < 5]
		X_test_0_4 = X_test_full[y_test_full < 5]
		y_test_0_4 = y_test_full[y_test_full < 5]
		return X_train_0_4, y_train_0_4, X_test_0_4, y_test_0_4

	def get_train_and_test_data_5_9(self):
		X_train_full, y_train_full, X_test_full, y_test_full = self.get_full_train_and_test_data()
		X_train_5_9 = X_train_full[y_train_full > 4]
		X_test_5_9 = X_test_full[y_test_full > 4]

		# pre trained model expects labels in range 0-4
		# so do "mapper" for new labels
		y_train_5_9 = y_train_full[y_train_full > 4] - 5
		y_test_5_9 = y_test_full[y_test_full > 4] - 5

		return X_train_5_9, y_train_5_9, X_test_5_9, y_test_5_9

	@staticmethod
	def get_full_train_and_test_data():
		(X_train, y_train), (X_test, y_test) = mnist.load_data()
		X_train, X_test = X_train / 255, X_test / 255
		return X_train, y_train, X_test, y_test

	@staticmethod
	def show_image_from_data_set(arr_digits_from_data_set):
		digit_size = (8, 8)
		img_per_row = 5
		num_images = len(arr_digits_from_data_set)
		num_plt_rows = num_images // img_per_row

		# for situation when images number isn't match with number plt columns
		temp = num_images % img_per_row
		if temp != 0:
			num_plt_rows = num_plt_rows + 1
		fig, axs = plt.subplots(num_plt_rows, img_per_row, figsize = digit_size)

		# turn off axis for all subplots
		[axi.set_axis_off() for axi in axs.ravel()]

		for img_idx in range(num_images):
			row_idx = img_idx // img_per_row
			col_idx = img_idx - row_idx*img_per_row
			axs[row_idx, col_idx].imshow(arr_digits_from_data_set[img_idx], cmap='Greys')

		plt.show()

	@staticmethod
	def show_one_img(digit):
		plt.imshow(digit, cmap='Greys')
		plt.show()
