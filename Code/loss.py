import tensorflow as tf

def discriminator_loss(logits_real, logits_fake):

    loss = None

    lossfunc = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = lossfunc(tf.ones(logits_real.shape),logits_real)
    fake_loss = lossfunc(tf.zeros(logits_fake.shape),logits_fake)
    loss = real_loss + fake_loss

    return loss

def generator_loss(logits_fake):

    loss = None
    lossfunc = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    loss = lossfunc(tf.ones(logits_fake.shape),logits_fake)

    return loss
