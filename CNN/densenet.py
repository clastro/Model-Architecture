from tensorflow.keras import datasets, layers, models, Model, optimizers, metrics
from tensorflow.keras.layers import InputLayer, Dense, Flatten, Reshape, Input, Conv1D, BatchNormalization, ReLU, MaxPool1D, concatenate, AvgPool1D, GlobalAveragePooling1D

def conv_layer(x, filters, kernel=1, strides=1):
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv1D(filters, kernel, strides=strides, padding="same")(x)
    return x

def transition_layer(x):
    x = conv_layer(x, x.shape[-1] // 2)
    x = AvgPool1D(2, strides=2, padding="same")(x)
    return x
    
def dense_block(x, repetition, filters):
    for _ in range(repetition):
        y = conv_layer(x, 4 * filters)
        y = conv_layer(y, filters, 3)
        x = concatenate([y, x])
    return x
    
def densenet(input_shape, n_classes, filters=32):

    input = Input(input_shape)
    x = Conv1D(64, 7, strides=2, padding="same")(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool1D(3, strides=2, padding="same")(x)

    for repetition in [6, 12, 24, 16]:

        d = dense_block(x, repetition, filters)
        x = transition_layer(d)
    x = GlobalAveragePooling1D()(d)
    output = Dense(n_classes, activation="softmax")(x)

    model = Model(input, output)
    return model

