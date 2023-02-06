import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import tensorflow.keras as keras
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
def get_cifar10():
    img_rows = 32
    img_cols = 32
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
    num_category = 10
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_category)
    y_test = keras.utils.to_categorical(y_test, num_category)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    return (X_train, y_train, X_test, y_test)




def accuracy(y_true, y_pred):
    correct_num = tf.equal(tf.argmax(y_true, -1), tf.argmax(y_pred, -1))
    accuracy = tf.reduce_mean(tf.cast(correct_num, dtype=tf.float32))
    return accuracy

from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow import keras
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import Model

def get_data(task):
    img_size = 32

    if task == 'cifar10':
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        X_train = X_train.reshape(X_train.shape[0], img_size, img_size, 3)

        X_test = X_test.reshape(X_test.shape[0], img_size, img_size, 3)

        num_category = 10
        # convert class vectors to binary class matrices

        # y_train = keras.utils.to_categorical(y_train, num_category)

        # y_test = keras.utils.to_categorical(y_test, num_category)

        X_train = X_train.astype('float32')

        X_test = X_test.astype('float32')

        X_train /= 255

        X_test /= 255

    elif task=='cifar100':

        (X_train, y_train), (X_test, y_test) = cifar100.load_data()
        X_train = X_train.reshape(X_train.shape[0], img_size, img_size, 3)

        X_test = X_test.reshape(X_test.shape[0], img_size, img_size, 3)

        num_category = 100
        # convert class vectors to binary class matrices

        y_train = keras.utils.to_categorical(y_train, num_category)

        y_test = keras.utils.to_categorical(y_test, num_category)

        X_train = X_train.astype('float32')

        X_test = X_test.astype('float32')

        X_train /= 255

        X_test /= 255




    return (X_train,y_train),(X_test,y_test)

def AUROC_Score(pred_in, pred_out,file):

    y_in = [1]*len(pred_in)
    y_out = [0]*len(pred_out)

    y = y_in + y_out

    pred = pred_in.tolist() + pred_out.tolist()
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    plt.plot(fpr, tpr, label=file)
    plt.savefig(file+".png",bbox_inches='tight')
    print(roc_auc_score(y, pred))




class CIFARModel:

    def __init__(self, restore,session=None):


        model=keras.models.load_model(restore)



        self.model=model









    def summary(self):

        print(self.model.summary())

    def ft(self,data):
        out = self.model.layers[-3].output
        model_ft = Model(self.model.layers[0].input, out)

        return (model_ft(data))

    def predict(self,data):

        out = self.model.layers[-2].output

        model_new = Model(self.model.layers[0].input, out)

        return (model_new(data))

    def give(self,black):
        out = self.model.layers[-1].output
        model_new = Model(self.model.layers[0].input, out)
        return (model_new.predict(black))


