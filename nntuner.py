from itertools import permutations
import numpy as np
from sknn.mlp import Classifier, Layer
from sklearn.cross_validation import train_test_split


def load_data(file_path):
    return np.genfromtxt(file_path, delimiter=',', missing_values='?',
                         dtype=None)

def split_set(data):
    return (data[:,:-1], data[:,-1])


data = load_data('dermatology/dermatology.data')
layers = [Layer("Rectifier", units=100, dropout=0.25),
          Layer("Linear", units=100, dropout=0.25),]
          #Layer("Sigmoid", units=100, dropout=0.25)]

for layers_perm in permutations(layers):
    X_train, X_test, y_train, y_test = train_test_split(data[:, :-1],
                                                        data[:, -1],
                                                        test_size=0.33,
                                                        random_state=42)
    nn = Classifier(
        layers=list(layers_perm) + [Layer("Softmax")],
        learning_rate=0.001,
        n_iter=100)

    nn.fit(X_train, y_train)
    print "Configuration: " + str(layers_perm)
    print nn.score(X_test, y_test)
