#!/usr/bin/env python3
"""Something that function does"""


from tensorflow import keras as K

identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Builds the ResNet-50 architecture.
    Returns:
        keras.Model: the ResNet-50 keras model
    """
    init = K.initializers.HeNormal(seed=0)
    X = K.Input(shape=(224, 224, 3))
    # Stage 1: conv1
    Y = K.layers.Conv2D(64, 7, strides=2, padding='same',
                        kernel_initializer=init)(X)
    Y = K.layers.BatchNormalization(axis=3)(Y)
    Y = K.layers.ReLU()(Y)
    Y = K.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(Y)
 
    # Stage 2: conv2_x — projection + 2 identity blocks, filters=[64,64,256]
    Y = projection_block(Y, [64, 64, 256], s=1)
    Y = identity_block(Y, [64, 64, 256])
    Y = identity_block(Y, [64, 64, 256])
    # Stage 3: conv3_x — projection + 3 identity blocks, filters=[128,128,512]
    Y = projection_block(Y, [128, 128, 512], s=2)
    Y = identity_block(Y, [128, 128, 512])
    Y = identity_block(Y, [128, 128, 512])
    Y = identity_block(Y, [128, 128, 512])
    # Stage 4: conv4_x — projection + 5 identity blocks, filters=[256,256,1024]
    Y = projection_block(Y, [256, 256, 1024], s=2)
    Y = identity_block(Y, [256, 256, 1024])
    Y = identity_block(Y, [256, 256, 1024])
    Y = identity_block(Y, [256, 256, 1024])
    Y = identity_block(Y, [256, 256, 1024])
    Y = identity_block(Y, [256, 256, 1024])
    # Stage 5: conv5_x — projection + 2 identity blocks, filters=[512,512,2048]
    Y = projection_block(Y, [512, 512, 2048], s=2)
    Y = identity_block(Y, [512, 512, 2048])
    Y = identity_block(Y, [512, 512, 2048])
    # Average pooling + fully connected
    Y = K.layers.AveragePooling2D(pool_size=7, strides=1)(Y)
    Y = K.layers.Dense(1000, activation='softmax',
                       kernel_initializer=init)(Y)
    return K.Model(inputs=X, outputs=Y)
