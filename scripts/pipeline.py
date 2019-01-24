from scripts.data import MNISTData
from scripts.model import DigitClassifier


#  --------------
# Step 1
# getting data and familiarity with it

data_obj = MNISTData()
X_train_0_4, y_train_0_4, X_test_0_4, y_test_0_4 = data_obj.get_train_and_test_data_0_4()

# visual familiarity with data
# is in data sets only right digits
# digit = X_train_0_4[0:35]
# data_obj.show_image_from_data_set(digit)

#  --------------
# Step 2
# training model
clf_o = DigitClassifier()
# clf_o.get_best_parameters(X_train_0_4, y_train_0_4)
# model_1 = clf_o.build_model()
# model_1 = clf_o.train_model(model_1, X_train_0_4, y_train_0_4)
# clf_o.evaluate_model(model_1, X_test_0_4, y_test_0_4)


#  --------------
# Step 3
# saving model
# clf_o.save_model(model_1, 'model 1')


#  --------------
# Step 4
# restoring model
model_1 = clf_o.load_my_model('model 1')


#  --------------
# Step 5
# rebuilding loaded model
# clf_o.evaluate_model(model_1, X_test_0_4, y_test_0_4)
model_2 = clf_o.rebuild_model(model_1)
X_train_5_9, y_train_5_9, X_test_5_9, y_test_5_9 = data_obj.get_train_and_test_data_5_9()
model_2 = clf_o.train_model(model_2, X_train_5_9, y_train_5_9)
# clf_o.save_model(model_2, 'model 2')
# clf_o.evaluate_model(model_2, X_test_5_9, y_test_5_9)


#  --------------
# Step 6
# predict testing
# X_train_5_9, y_train_5_9, X_test_5_9, y_test_5_9 = data_obj.get_train_and_test_data_5_9()
# model_2 = clf_o.load_my_model('model 2')

# data_obj.show_one_img(X_test_5_9[0])
# clf_o.predict_number(model_2, X_test_5_9[0], y_test_5_9[0])
#
# data_obj.show_one_img(X_test_5_9[10])
# clf_o.predict_number(model_2, X_test_5_9[10], y_test_5_9[10])
