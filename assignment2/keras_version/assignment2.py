from keras.models import Sequential
from keras.layers import Embedding, Dense, Merge
from keras.layers.core import Flatten, Reshape
from keras.optimizers import SGD
from keras import initializations
import scipy.io as sio
import numpy as nb
import numpy.matlib as ml

batchsize = 100  # Mini-batch size.
learning_rate = 0.1  # Learning rate; default = 0.1.
momentum = 0.9  # Momentum; default = 0.9.
embedding_dims = 50  # Dimensionality of embedding space; default = 50.
hidden_unit_count = 200  # Number of units in hidden layer; default = 200.
init_wt = 0.01  # Standard deviation of the normal distribution for initial weights; default = 0.01

mat_contents = sio.loadmat('/Users/remington/Pillar/coursera-neural-networks/assignment2/data.mat')
data = mat_contents['data']

train_data = data['trainData'][0,0]
test_data = data['testData'][0,0]
valid_data = data['validData'][0,0]
vocabulary = data['vocab'][0,0]
vocabulary_size = vocabulary.size

def convertToOneOfK(indices, size_of_indices):
    returnValue = nb.zeros((indices.size, size_of_indices))
    returnValue[nb.arange(indices.size), indices] = 1
    return returnValue

inputs = train_data.transpose()[:, :3] - 1 # dims = (:, 3)
labels = convertToOneOfK(train_data.transpose()[:, 3] - 1, vocabulary_size) # dims = (:, 250)

def custom_init(shape, name=None):
    return initializations.normal(shape, scale=init_wt, name=name)

model = Sequential()
model.add(Embedding(vocabulary_size, embedding_dims, input_length=3, init=custom_init))
model.add(Flatten())
model.add(Dense(hidden_unit_count, init=custom_init, activation='sigmoid'))
model.add(Dense(vocabulary_size, init=custom_init, activation='softmax'))
model.summary()

optimizer = SGD(lr=learning_rate, momentum=momentum)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(inputs, labels, batch_size=batchsize, nb_epoch=5)

def display_nearest_words(model, vocabulary, word):
    index = nb.argwhere(vocabulary == word)[0,1]
    embedding_weights = model.layers[0].get_weights()[0]
    target_word_weights = embedding_weights[index, :]
    diffs = embedding_weights - ml.repmat(target_word_weights,  vocabulary.size, 1)
    distances = nb.sqrt(nb.sum(diffs * diffs, axis=1))
    indices_by_distance = nb.argsort(distances)
    for index in nb.nditer(indices_by_distance):
        print(vocabulary[0,index][0])
