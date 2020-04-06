import tensorflow as tf

NOISE_DIM = 96
def discriminator():

    model = tf.keras.models.Sequential([

        tf.keras.layers.InputLayer((784,)),
        tf.keras.layers.Reshape((28,28,1)),
        tf.keras.layers.Conv2D(filters=32,kernel_size=[5,5],strides=1,padding='valid',input_shape=(28,28,1)),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
        tf.keras.layers.Conv2D(filters=64,kernel_size=[5,5],strides=1,padding='valid'),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4*4*64),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.Dense(1)

    ])
    return model


def generator(noise_dim=NOISE_DIM):

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(1024,activation='relu',input_shape=(noise_dim,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(7*7*128,activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Reshape((7,7,128)))
    model.add(tf.keras.layers.Conv2DTranspose(filters=64,kernel_size=[4,4],strides=2,padding='same',activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2DTranspose(filters=1,kernel_size=[4,4],strides=2,padding='same',activation='tanh'))

    return model
