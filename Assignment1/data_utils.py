from pickletools import read_unicodestringnl
import numpy as np
from urllib import request
import gzip
import pickle
 
filename = [
["training_images","train-images-idx3-ubyte.gz"],
["test_images","t10k-images-idx3-ubyte.gz"],
["training_labels","train-labels-idx1-ubyte.gz"],
["test_labels","t10k-labels-idx1-ubyte.gz"]
]
 
def download_mnist():
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        print("Downloading "+name[1]+"...")
        request.urlretrieve(base_url+name[1], name[1])
    print("Download complete.")
 
def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("mnist.pkl", 'wb') as f:
        pickle.dump(mnist,f)
    print("Save complete.")
 
def init_mnist():
    download_mnist()
    save_mnist()
 
def load_mnist():
    with open("mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
    sample_array = np.random.choice(60000, 60000)
    mnist['training_images'] = mnist['training_images'][sample_array, :]
    mnist['training_labels'] = mnist['training_labels'][sample_array]
    data = {}
    data['X_train'] = mnist['training_images'][range(55000), :]
    data['y_train'] = mnist['training_labels'][range(55000)]
    data['X_val'] = mnist['training_images'][range(55000, 60000), :]
    data['y_val'] = mnist['training_labels'][range(55000, 60000)]
    data['X_test'] = mnist['test_images']
    data['y_test'] = mnist['test_labels']
    return data
 
if __name__ == '__main__':
    init_mnist()

