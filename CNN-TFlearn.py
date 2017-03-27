import tflearn
import tflearn.datasets.mnist as mnist
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

## mnist dataset - 55000 samples
X, Y, test_x, test_y = mnist.load_data(one_hot=True)
print(test_x)

# conv_2d
# INPUT  4-D Tensor [batch, height, width, in_channels = columns].
# Output 4-D Tensor [batch, new height, new width, nb_filter].

X = X.reshape([55000, 28, 28, 1])

test_x = test_x.reshape([10000, 28, 28, 1])

CNN = input_data(shape=[None, 28, 28, 1], name='input')

#         incoming, #nb_filter, #Size of filters
CNN = conv_2d(CNN, 32, 2, activation='relu')
#            incoming, #nb_filter, #kernel_size
CNN = max_pool_2d(CNN, 2)

CNN = conv_2d(CNN, 64, 2, activation='relu')
CNN = max_pool_2d(CNN, 2)

#                       incoming,  n_units
CNN = fully_connected(CNN, 1024, activation='relu')
CNN = dropout(CNN, 0.80)

CNN = fully_connected(CNN, 10, activation='softmax')
CNN = regression(CNN, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(CNN)
model.fit({'input': X}, {'targets': Y}, n_epoch=10, validation_set=({'input': test_x}, {'labels': test_y}),
	snapshot_step=750, show_metric=True, run_id='mnist')

#model.save('CNN.model')

#model.load('CNN.model')
