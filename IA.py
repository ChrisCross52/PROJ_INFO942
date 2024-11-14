import numpy as np
from numpy import asarray
from PIL import Image
import os
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

# Configuration du chemin des images et de la taille d'image
image_path = '/workspaces/PROJ_INFO942/Images/out2'

# Fonction de chargement et de normalisation des images
def getImage(path, image_size=(64, 64)):
    images = []
    for img in os.listdir(path):
        temp_image = Image.open(os.path.join(path, img))
        temp_image = temp_image.convert('RGB')
        image = temp_image.resize(image_size)
        image = np.asarray(image)
        image = ((image - 127.5) / 127.5).astype("float32")
        images.append(image)
    return np.asarray(images)

train_images = getImage(image_path)
print(train_images.shape)

# Paramètres du modèle
LATENT_DIM = 100
WEIGHT_INIT = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
CHANNELS = 3  # Pour une image en couleur

# Modèle du générateur
def generator_model():
    model = keras.Sequential(name='generator')
    model.add(layers.Dense(8 * 8 * 512, input_dim=LATENT_DIM))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Reshape((8, 8, 512)))
    model.add(layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=WEIGHT_INIT))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=WEIGHT_INIT))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=WEIGHT_INIT))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2DTranspose(CHANNELS, (4, 4), padding='same', activation='tanh'))
    return model

generator = generator_model()
generator.summary()

# Modèle du discriminateur
def disc_model():
    model = keras.Sequential(name='discriminator')
    input_shape = (64, 64, 3)
    alpha = 0.2
    model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=alpha))
    model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=alpha))
    model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=alpha))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

discriminator = disc_model()
discriminator.summary()

# Création du dossier pour sauvegarder les images générées
output_folder = 'ImagesGénérées'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Classe pour surveiller et sauvegarder les images générées
class DCGANMonitor(keras.callbacks.Callback):
    def __init__(self, num_imgs=25, latent_dim=100):
        self.num_imgs = num_imgs
        self.latent_dim = latent_dim
        self.noise = tf.random.normal([self.num_imgs, latent_dim])

    def on_epoch_end(self, epoch, logs=None):
        generated_images = self.model.generator(self.noise, training=False)
        generated_images = (generated_images * 127.5 + 127.5).numpy()  # Dé-normalisation

        fig = plt.figure(figsize=(8, 8))
        for i in range(self.num_imgs):
            plt.subplot(5, 5, i + 1)
            img = array_to_img(generated_images[i])
            plt.imshow(img)
            plt.axis('off')
        
        output_path = os.path.join(output_folder, f'epoch_{epoch + 1:03d}.png')
        fig.savefig(output_path)
        plt.close(fig)

    def on_train_end(self, logs=None):
        print("Entraînement terminé, les images générées ont été sauvegardées.")

# Définition de la classe GAN et compilation
class DCGAN(keras.Model):
    def __init__(self, generator, discriminator, latent_dim):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.g_loss_metric = keras.metrics.Mean(name='g_loss')
        self.d_loss_metric = keras.metrics.Mean(name='d_loss')

    @property
    def metrics(self):
        return [self.g_loss_metric, self.d_loss_metric]

    def compile(self, g_optimizer, d_optimizer, loss_fn):
        super(DCGAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        random_noise = tf.random.normal(shape=(batch_size, self.latent_dim))

        with tf.GradientTape() as tape:
            pred_real = self.discriminator(real_images, training=True)
            real_labels = tf.ones((batch_size, 1)) + 0.05 * tf.random.uniform(tf.shape(real_labels))
            d_loss_real = self.loss_fn(real_labels, pred_real)

            fake_images = self.generator(random_noise)
            pred_fake = self.discriminator(fake_images, training=True)
            fake_labels = tf.zeros((batch_size, 1))
            d_loss_fake = self.loss_fn(fake_labels, pred_fake)
            d_loss = (d_loss_real + d_loss_fake) / 2

        gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))

        labels = tf.ones((batch_size, 1))
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_noise, training=True)
            pred_fake = self.discriminator(fake_images, training=True)
            g_loss = self.loss_fn(labels, pred_fake)

        gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        
        return {'d_loss': self.d_loss_metric.result(), 'g_loss': self.g_loss_metric.result()}

dcgan = DCGAN(generator=generator, discriminator=discriminator, latent_dim=LATENT_DIM)
D_LR = 0.0001
G_LR = 0.0003
dcgan.compile(g_optimizer=Adam(learning_rate=G_LR, beta_1=0.5), d_optimizer=Adam(learning_rate=D_LR, beta_1=0.5), loss_fn=BinaryCrossentropy())
N_EPOCHS = 45
dcgan.fit(train_images, epochs=N_EPOCHS, callbacks=[DCGANMonitor()])
