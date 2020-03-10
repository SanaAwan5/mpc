import numpy as np
from datetime import datetime

x_train_features = np.load('x_train_features.npy')
y_train = np.load('y_train.npy')

x_test_features = np.load('x_test_features.npy')
y_test = np.load('y_test.npy')

print(x_train_features.shape, y_train.shape, x_test_features.shape, y_test.shape)

from pond.tensor import NativeTensor, PrivateEncodedTensor, PublicEncodedTensor
from pond.nn import Dense, Sigmoid, Reveal, Diff, Softmax, CrossEntropy, Sequential, DataLoader

classifier = Sequential([
    Dense(128, 4732),
    Sigmoid(),
    # Dropout(.5),
    Dense(10, 128),
    Reveal(),
    Softmax()
])

def accuracy(classifier, x, y, verbose=0, wrapper=NativeTensor):
    predicted_classes = classifier \
        .predict(DataLoader(x, wrapper), verbose=verbose).reveal() \
        .argmax(axis=1)
        
    correct_classes = NativeTensor(y) \
        .argmax(axis=1)
        
    matches = predicted_classes.unwrap() == correct_classes.unwrap()
    return sum(matches)/len(matches)

classifier.initialize(x_train_features.shape, PrivateEncodedTensor)

start = datetime.now()
classifier.fit(
    DataLoader(x_train_features, wrapper=PrivateEncodedTensor), 
    DataLoader(y_train, wrapper=PrivateEncodedTensor),
    loss=CrossEntropy(), 
    epochs=3,
    learning_rate=.01,
    verbose=1
)
stop = datetime.now()

print("Elapsed:", stop - start)