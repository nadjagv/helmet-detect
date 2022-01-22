import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dense, Flatten


def VGG16_D(num_classes, input_shape=(224, 224, 3), pretrained=True):
    """ Instantiate a Keras VGG16 configuration D model. The name of each layer follows Keras's official implementation so that
    the pretrained weights on ImageNet offered by Keras can be loaded easily.

    Args:
        - num_classes: The number of classes in the dataset.
        - input_shape: The shape of the input image. Defaults to (224, 224, 3).

    Returns:
        - A Keras model instance.

    Code References:
        - https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg16.py

    Paper References:
        - Simonyan, K., & Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition.
          International Conference on Learning Representation (ICLR) 2015. https://arxiv.org/abs/1409.1556
    """
    input_layer = Input(shape=input_shape, name="input_1")
    block1_conv1 = Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same', name='block1_conv1')(input_layer)
    block1_conv2 = Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same', name='block1_conv2')(block1_conv1)
    block1_pool = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same", name="block1_pool")(block1_conv2)
    block2_conv1 = Conv2D(128, kernel_size=(3, 3), activation="relu", padding='same', name='block2_conv1')(block1_pool)
    block2_conv2 = Conv2D(128, kernel_size=(3, 3), activation="relu", padding='same', name='block2_conv2')(block2_conv1)
    block2_pool = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same", name="block2_pool")(block2_conv2)
    block3_conv1 = Conv2D(256, kernel_size=(3, 3), activation="relu", padding='same', name='block3_conv1')(block2_pool)
    block3_conv2 = Conv2D(256, kernel_size=(3, 3), activation="relu", padding='same', name='block3_conv2')(block3_conv1)
    block3_conv3 = Conv2D(256, kernel_size=(3, 3), activation="relu", padding='same', name='block3_conv3')(block3_conv2)
    block3_pool = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same", name="block3_pool")(block3_conv3)
    block4_conv1 = Conv2D(512, kernel_size=(3, 3), activation="relu", padding='same', name='block4_conv1')(block3_pool)
    block4_conv2 = Conv2D(512, kernel_size=(3, 3), activation="relu", padding='same', name='block4_conv2')(block4_conv1)
    block4_conv3 = Conv2D(512, kernel_size=(3, 3), activation="relu", padding='same', name='block4_conv3')(block4_conv2)
    block4_pool = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same", name="block4_pool")(block4_conv3)
    block5_conv1 = Conv2D(512, kernel_size=(3, 3), activation="relu", padding='same', name='block5_conv1')(block4_pool)
    block5_conv2 = Conv2D(512, kernel_size=(3, 3), activation="relu", padding='same', name='block5_conv2')(block5_conv1)
    block5_conv3 = Conv2D(512, kernel_size=(3, 3), activation="relu", padding='same', name='block5_conv3')(block5_conv2)
    block5_pool = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same", name="block5_pool")(block5_conv3)
    flatten = Flatten(name="flatten")(block5_pool)
    fc1 = Dense(4096, activation="relu", name="fc1")(flatten)
    fc2 = Dense(4096, activation="relu", name="fc2")(fc1)
    predictions = Dense(num_classes, activation="softmax", name="predictions")(fc2)
    model = Model(inputs=input_layer, outputs=predictions)

    if pretrained:
        weight_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
        )
        model.load_weights(weight_path, by_name=True)

    return model
