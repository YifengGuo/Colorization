from activators import SigmoidActivator, ReluActivator
import numpy as np
from datetime import datetime


class FullyConnectedLayer(object):
    def __init__(self, input_size, output_size, activator):
        '''
        constructor
        :param input_size: 
        :param output_size: 
        :param activator: 
        '''
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator

        # vector of weights W
        # if current layer has 3 neurons and next layer has 2 neurons
        # the weight vector look like:
        # [[w41, w42, w43],
        #  [w51, w52, w53]]
        # and input_size = # of neurons in current layer
        # looks like:
        #        [[x1],
        #         [x2],
        #         [x3]]
        # and the output = W * x + bi
        # self.W = np.random.uniform(
        #     -000.1, 000.1,
        #     (output_size, input_size)
        # )
        self.W = np.zeros((output_size, input_size))
        # bias for each output term
        # np.zeros((x, y)) return x * y dimension matrix of zeros
        self.b = np.zeros((output_size, 1))

        # output
        self.output = np.zeros((output_size, 1))

    def forward(self, input_array):
        '''
        forward calculation for current layer
        :param input_array: vector of input, its dimension must be equal to input_size 
        :return: 
        '''
        self.input = input_array
        # weighted_input = np.dot(self.W, input_array)
        # self.output = self.activator.forward(weighted_input) + self.b
        self.output = self.activator.forward(
            np.dot(self.W, input_array) + self.b)

    def backward(self, delta_array):
        '''
        backpropagation to calculate gradient of W and b
        :param delta_array: the vector delta_j calculated and sent from the next layer
        :return: 
        '''
        # calculate current layer's delta
        # for hidden layer:
        # delta_i = ai * (1 - ai) * sigma(Wki * delta_k)
        # ai is i-th layer output
        # in network calc_gradient(), we first calculate the delta of output layer by (tj - yj) * yj * (1 - yj)
        # then from last layer to first layer run backward()
        # so ith input is i - 1th layer's output, and the delta of output layer is pre-calculated
        # so what we do in below is to calculate the delta of the previous layer
        self.delta = self.activator.backward(self.input) * np.dot(self.W.T, delta_array)
        self.W_gradient = np.dot(delta_array, self.input.T)
        self.b_gradient = delta_array

    def update(self, learning_rate):
        self.W += learning_rate * self.W_gradient
        self.b += learning_rate * self.b_gradient


class Network(object):
    def __init__(self, layers):
        '''
        constructor
        :param layers: 
        '''
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(FullyConnectedLayer(
                layers[i],
                layers[i + 1],
                SigmoidActivator()
            ))

    def predict(self, sample):
        '''
        predict label by neural network given sample
        run activator on input and the output of current layer is the input of next layer
        :param sample: input sample
        :return: 
        '''
        output = sample
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output

    def calc_gradient(self, label):
        '''
        gradient = delta(Ed) / (delta Wji) = (delta Ed / delta net_j) * Xji = delta * Xji
        and (delta Ed / delta net_j) = delta sent from next layer
        Xji is the output of current layer and input out the next layer
        :param label: 
        :return: 
        '''
        # first calculate delta of output layer
        # delta = yj*(1 - yj)*(tj - yj)
        delta = self.layers[-1].activator.backward(self.layers[-1].output) * (label - self.layers[-1].output)
        # then calculate delta transmitted within hidden layers
        for layer in self.layers[::-1]:  # from last to the first layer transmit delta
            layer.backward(delta)  # calculate gradient for connection and bias on this layer
            delta = layer.delta
        return delta

    def update_weight(self, rate):
        for layer in self.layers:
            layer.update(rate)

    def train(self, labels, samples, rate, epoch):
        for i in range(epoch):
            for j in range(len(samples)):
                self.train_one_sample(labels[j], samples[j], rate)

    def train_one_sample(self, label, sample, rate):
        self.predict(sample)
        self.calc_gradient(label)
        self.update_weight(rate)

    def loss(self, label, output):
        res = 0.5 * (label - output) * (label - output).sum()
        return np.sum(res)


def get_train_data_set():
    train_input = []
    train_labels = []
    fileIn1 = open('input_40000.csv')
    for line in fileIn1.readlines():
        lineArr = line.strip().split(',')
        train_input.append(map(float, lineArr))

    fileIn2 = open('color_40000_labels.csv')
    for line in fileIn2.readlines():
        lineArr = line.strip().split(',')
        train_labels.append(map(float, lineArr))

    fileIn1.close()
    fileIn2.close()
    return train_input, train_labels


def get_t_data_set():
    '''
    test data set
    :return: 
    '''
    t_input = []
    t_labels = []
    fileIn1 = open('input_test.csv')
    for line in fileIn1.readlines():
        lineArr = line.strip().split(',')
        t_input.append(map(float, lineArr))

    fileIn2 = open('color_labels_test.csv')
    for line in fileIn2.readlines():
        lineArr = line.strip().split(',')
        t_labels.append(map(float, lineArr))

    fileIn1.close()
    fileIn2.close()
    return t_input, t_labels


def transpose(args):
    return map(
        lambda arg: map(
            lambda line: np.array(line).reshape(len(line), 1)
            , arg)
        , args
    )


def normalize_labels_train():
    '''
    transform labels to one-hot vector
    :return: 
    '''
    labels_index = []
    clustering_labels = []
    dict = {}
    train_input, train_labels = get_train_data_set()
    fileIn = open('labels.csv')
    for line in fileIn.readlines():
        lineArr = line.strip().split(',')
        clustering_labels.append([float(lineArr[0]), float(lineArr[1]), float(lineArr[2])])
    fileIn.close()

    for i in range(len(clustering_labels)):
        dict[i] = clustering_labels[i]

    for label in train_labels:
        for j in range(len(dict)):
            if dict[j] == label:
                labels_index.append(j)

    one_hot_vector = []
    for i in range(len(labels_index)):
        one_hot_vector.append(padding(labels_index[i]))

    return one_hot_vector


def normalize_labels_t():
    '''
    for test
    transform labels to one-hot vector
    :return: 
    '''
    labels_index = []
    clustering_labels = []
    dict = {}
    test_input, test_labels = get_t_data_set()
    fileIn = open('labels.csv')
    for line in fileIn.readlines():
        lineArr = line.strip().split(',')
        clustering_labels.append([float(lineArr[0]), float(lineArr[1]), float(lineArr[2])])
    fileIn.close()

    for i in range(len(clustering_labels)):
        dict[i] = clustering_labels[i]

    for label in test_labels:
        for j in range(len(dict)):
            if dict[j] == label:
                labels_index.append(j)

    one_hot_vector = []
    for i in range(len(labels_index)):
        one_hot_vector.append(padding(labels_index[i]))

    return one_hot_vector


def padding(index):
    '''
    construct one hot vector
    the element at index is 0.9
    the others are 0.1
    :param index: 
    :return: 
    '''
    res = []
    for i in range(6):
        if i == index:
            res.append(0.9)
        else:
            res.append(0.1)
    return res


def aux(arg):
    '''
    transpose n * m matrix to n (m * 1) matrices
    :param arg: 
    :return: 
    '''
    return map(
        lambda line: np.array(line).reshape(len(line), 1)
        , arg)


def get_max_index(vec):
    '''
    in hot one vector, only one element is 0.9, the others are all 0.1
    the result of neural network should be the clustering label the index with value 0.9 corresponding to
    :param vec: 
    :return:  index with value 0.9
    '''
    max_value_index = 0
    max_value = 0
    for i in range(len(vec)):
        if vec[i] > max_value:
            max_value = vec[i]
            max_value_index = i
        else:
            continue
    return max_value_index


def evaluate(network, test_input, test_labels):
    '''
    error = (error count) / total count
    error will be increase by one if result of NN prediction does not equal result from label
    :param network: 
    :param test_input: 
    :param test_labels: 
    :return: 
    '''
    error = 0
    total = len(test_input)

    for i in range(total):
        prediction = get_max_index(network.predict(test_input[i]))
        label = get_max_index(test_labels[i])
        if label != prediction:
            error += 1
    return float(error) / float(total)


def train_evaluate():
    last_error_ratio = 1.0
    epoch = 0
    train_input, train_labels = transpose(get_train_data_set())
    train_labels = aux(normalize_labels_train())
    test_input, test_labels = transpose(get_t_data_set())
    test_labels = aux(normalize_labels_t())
    network = Network([9, 300, 6])
    # print network.loss(train_labels[-1], network.predict(train_input[-1]))
    while True:
        epoch += 1
        network.train(train_labels, train_input, 0.08, 1)
        print '%s epoch %d finished, loss: %f' % (str(datetime.now()), epoch,
                                                  network.loss(train_labels[-1], network.predict(train_input[-1])))
        if epoch % 2 == 0:
            current_error_ratio = evaluate(network, test_input, test_labels)
            print '%s, after %d epoch, current error ratio is: %f' % (str(datetime.now()), epoch, current_error_ratio)
            if current_error_ratio > last_error_ratio:
                break
            else:
                last_error_ratio = current_error_ratio

        if epoch > 50:
            break
    return network


def get_new_data():
    new_data = []
    fileIn = open('data.csv')
    for line in fileIn.readlines():
        lineArr = line.strip().split(',')
        new_data.append(map(float, lineArr))
    fileIn.close()
    return new_data


def decode_index(list):
    '''
    decode index to corresponding rgb labels
    :return: 
    '''
    dict = {}
    labels = []
    fileIn = open('labels.csv')
    for line in fileIn.readlines():
        lineArr = line.strip().split(',')
        labels.append(map(float, lineArr))
    fileIn.close()

    for i in range(len(labels)):
        dict[i] = labels[i]

    color = []
    for i in range(len(list)):
        color.append(dict[list[i]])
    return color

import csv


def list_to_csv(list):
    with open('data_color.csv', 'wb') as myfile:
        for line in list:
            myfile.writelines(','.join(map(repr, line)))
            myfile.write('\n')


from PIL import Image


def draw_image():
    color_list = []
    fileIn = open('data_color.csv')
    for line in fileIn.readlines():
        lineArr = line.strip().split(',')
        color_list.append(tuple(map(int, map(float, lineArr))))

    for t in color_list:
        print t
        # print type(tuple)

    im = Image.new("RGB", (481, 481))  # pixels of image
    pix = im.load()
    i = 0

    for x in range(481):
        for y in range(481):
            pix[x, y] = color_list[i]
            i += 1

    im.save("data_image.png", "PNG")
    print 'finished'


if __name__ == '__main__':
    # train_input, train_labels = get_train_data_set()
    # a = aux(normalize_labels())
    # print type(a[0])
    # for one_sample in train_input:
    #     print one_sample
    #
    # for one_label in train_labels:
    #     print one_label
    network = train_evaluate()
    input = get_new_data()
    # input = [[82, 82, 82, 82, 82, 82, 82, 82, 82],
    #          [82, 82, 81, 82, 82, 82, 82, 82, 82],
    #          [82, 81, 81, 82, 82, 82, 82, 82, 82],
    #          [81, 81, 82, 82, 82, 82, 82, 82, 82],
    #          [81, 82, 82, 82, 82, 82, 82, 82, 82]]
    list = []
    for i in range((len(input))):
        list.append(get_max_index(network.predict(aux(input)[i])))

    color = decode_index(list)
    for line in color:
        print line
    list_to_csv(color)

    draw_image()

    # a = normalize_labels()
    # b = np.array(a[:10])
    # print b.T





