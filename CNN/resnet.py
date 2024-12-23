from tensorflow.keras import Model
from tensorflow.keras.layers import InputLayer, Dense, Flatten, Input, Conv1D, BatchNormalization
from tensorflow.keras.layers import ReLU, Add, MaxPool1D, concatenate, GlobalAveragePooling1D

def basic_block(x, filters, kernel_size, strides = 1):
    
    x = Conv1D(filters = filters, kernel_size = kernel_size, strides = strides, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    return x

def identity_block(tensor, filters):
    
    x = basic_block(tensor, filters = filters, kernel_size =1, strides = 1)
    x = basic_block(x, filters = filters, kernel_size = 3, strides = 1)
    x = Conv1D(filters = 4 * filters, kernel_size = 1 , strides = 1)(x)
    x = BatchNormalization(x)
    x = Add()[tensor,x]) 
    x = ReLU()(x)

    return x

def convolutional_block(tensor, filters, strides):
    
    x = basic_block(tensor, filters=filters, kernel_size=1, strides=strides)
    x = basic_block(x, filters=filters, kernel_size=3, strides=1)
    x = Conv2D(filters=4*filters, kernel_size=1, strides=1)(x)
    x = BatchNormalization()(x)
    
    shortcut = Conv1D(filters= 4*filters, kernel_size=1, strides=strides)(tensor)
    shortcut = BatchNormalization()(shortcut)
    
    x = Add()([shortcut,x])    #skip connection
    x = ReLU()(x)

    return x
    
def resnet_block(x, filters, reps, strides):
    x = convolutional_block(x, filters, strides)
    for _ in range(reps-1):
        x = identity_block(x,filters)
    return x
    
def resnet(input_shape, n_classes):
    input = Input(input_shape)
    x = conv_batchnorm_relu(input, filters=64, kernel_size=7, strides=2)
    x = MaxPool1D(pool_size = 3, strides =2)(x)
    x = resnet_block(x, filters=64, reps =3, strides=1)
    x = resnet_block(x, filters=128, reps =4, strides=2)
    x = resnet_block(x, filters=256, reps =23, strides=2)
    x = resnet_block(x, filters=512, reps =3, strides=2)
    x = GlobalAvgPool1D()(x)
    
    output = Dense(n_classes, activation ='softmax')(x)
    model = Model(inputs=input, outputs=output)
    
    return model
