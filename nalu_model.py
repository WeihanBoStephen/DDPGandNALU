import numpy as np
import keras.backend as K
from keras.layers import *
from keras.initializers import *
from keras.models import *
import keras
import matplotlib.pyplot as plt
from keras.models import load_model
import opencv_jump as jump

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        #创建一个图
        plt.figure(figsize=(8,8))
        plt.title("NALU Training Loss")
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')#plt.plot(x,y)，这个将数据画成曲线
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        plt.grid(True)#设置网格形式
        plt.xlabel(loss_type)
        plt.ylabel('loss')#给x，y轴加注释
        plt.legend(loc="upper right")#设置图例显示位置
        plt.savefig("NALU_Train_Loss.jpg")
        plt.show()

class NALU(Layer):
    def __init__(self, units, MW_initializer='glorot_uniform',
                 G_initializer='glorot_uniform', mode="NALU",
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(NALU, self).__init__(**kwargs)
        self.units = units
        self.mode = mode
        self.MW_initializer = initializers.get(MW_initializer)
        self.G_initializer = initializers.get(G_initializer)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.W_hat = self.add_weight(shape=(input_dim, self.units),
                                     initializer=self.MW_initializer,
                                     name='W_hat')
        self.M_hat = self.add_weight(shape=(input_dim, self.units),
                                     initializer=self.MW_initializer,
                                     name='M_hat')
        if self.mode == "NALU":
            self.G = self.add_weight(shape=(input_dim, self.units),
                                     initializer=self.G_initializer,
                                     name='G')
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        W = K.tanh(self.W_hat) * K.sigmoid(self.M_hat)
        a = K.dot(inputs, W)
        if self.mode == "NAC":
            output = a
        elif self.mode == "NALU":
            m = K.exp(K.dot(K.log(K.abs(inputs) + 1e-7), W))
            g = K.sigmoid(K.dot(K.abs(inputs), self.G))
            output = g * a + (1 - g) * m
        else:
            raise ValueError("Valid modes: 'NAC', 'NALU'.")
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'mode' : self.mode,
            'MW_initializer': initializers.serialize(self.MW_initializer),
            'G_initializer':  initializers.serialize(self.G_initializer)
        }
        base_config = super(Dense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def get_history():
    return LossHistory()

def get_model(mode=NALU):
    x = Input((200,))
    y = NALU(2, mode=mode,
             MW_initializer=RandomNormal(stddev=1),
             G_initializer=Constant(10))(x)
    y = NALU(1, mode=mode,
             MW_initializer=RandomNormal(stddev=1),
             G_initializer=Constant(10))(y)
    return Model(x, y)

def load_model(flie_name):
    model = get_model("NALU")
    model.compile("rmsprop", "mse", metrics=["mae"])
    model.load_weights(flie_name)
    return model

def get_action(model,x):
    action = model.predict(x)[0][0]
    action += np.random.normal(0, 0.1)
    action = np.clip(action, -1, 1)
    action = (action + jump.get_action("autojump.jpg")) / 2.0
    return action

