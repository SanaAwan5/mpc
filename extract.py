import keras
import numpy as np
from dataset import load_mnist

n_train_items  = 640
n_test_items = 640
batch_size = 64

model = keras.models.load_model('pre-trained_model1.h5')
model.summary()

flatten_layer = model.get_layer(index=3)
assert flatten_layer.name.startswith('flatten_')

extractor = keras.models.Model(
    inputs=model.input, 
    outputs=flatten_layer.output
)

_, private_dataset = load_mnist()

(x_train_images, y_train), (x_test_images, y_test) = private_dataset


print(x_train_images.shape, y_train.shape, x_test_images.shape, y_test.shape) 

y_train = y_train[0:640]
y_test = y_test[0:640]   
x_train_images=x_train_images[0:640]  
x_test_images = x_test_images[0:640] 


x_train_features = extractor.predict(x_train_images[0:640])
x_test_features  = extractor.predict(x_test_images[0:640])

np.save('x_train_features.npy', x_train_features)
np.save('y_train.npy', y_train)

np.save('x_test_features.npy', x_test_features)
np.save('y_test.npy', y_test)