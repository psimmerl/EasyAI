"""Module holding classes for implementing (Varitational) AutoEncoders.
I stole most of this code from fchollet -- will rewrite eventually
https://keras.io/examples/generative/vae/ """
from typing import Union, Callable

import numpy as np
import tensorflow as tf

from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.layers import (Input, Dense, Dropout, BatchNormalization,
                                     LayerNormalization, LeakyReLU)
from tensorflow.keras.regularizers import L2
from tensorflow.keras.metrics import Mean


class Sampling(tf.keras.layers.Layer):
  """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

  def call(self, inputs):
    z_mean, z_log_var = inputs
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.random.normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VariationalAutoEncoder(tf.keras.Model):
  """Variational Autoencoder"""

  def __init__(self, n_feats, **kwargs):  #encoder, decoder
    super().__init__()#**kwargs)
    default_kwargs = {
      'latent_dim': 2,
      'variational' : True,
    }
    kwargs = default_kwargs | kwargs
    self.info = kwargs

    self.variational = self.info['variational']
    self.n_feats = n_feats
    self.latent_dim = self.info['latent_dim']
    self.total_loss_tracker = Mean(name='total_loss')
    self.reconstruction_loss_tracker = Mean(name='rec_loss')
    self.kl_loss_tracker = Mean(name='kl_loss')

    self.encoder = self.build_encoder()
    self.decoder = self.build_decoder()

    self.encoder.summary()
    self.decoder.summary()

  @property
  def metrics(self):
    return [
        self.total_loss_tracker,
        self.reconstruction_loss_tracker,
        self.kl_loss_tracker,
    ]

  def build_encoder(self):
    inputs_ = Input(shape=self.n_feats)
    for ilay, units in enumerate(4*[128]):#[256, 128, 64, 32]):
      x = Dense(
          units,
          activation='relu',
          # activation='linear,'
          kernel_initializer='he_normal',
      )(x if ilay else inputs_)
      # x = LeakyReLU(alpha=0.2)(x)
      # x = BatchNormalization()(x)
      x = LayerNormalization()(x)

    if self.variational:
      z_mean = Dense(self.latent_dim, name='z_mean')(x)
      z_log_var = Dense(self.latent_dim, name='z_log_var')(x)
      z = Sampling()([z_mean, z_log_var])

      return tf.keras.Model(inputs_, [z_mean, z_log_var, z], name='Encoder')
    else:
      z = Dense(self.latent_dim)(x)
      return tf.keras.Model(inputs_, z, name='Encoder')


  def build_decoder(self):
    inputs_ = Input(shape=self.latent_dim)
    for ilay, units in enumerate(4*[128]):#[32, 64, 128, 256]):
      x = Dense(
          units,
          activation='relu',
          # activation='linear',
          kernel_initializer='he_normal',
      )(x if ilay else inputs_)
      # x = LeakyReLU(alpha=0.2)(x)
      # x = BatchNormalization()(x)
      x = LayerNormalization()(x)

    x = Dense(self.n_feats, activation='linear')(x)

    return tf.keras.Model(inputs_, x, name='Decoder')

  def train_step(self, data):

    my_iter = tf.cast(self.optimizer.iterations, dtype=float)
    # kl_weight = min(1, self.my_iter/200000)
    # kl_weight = 0
    # if tf.reduce_all(my_iter > 10_000):
    kl_weight = tf.math.minimum(my_iter/300_000, 0.3)

    with tf.GradientTape() as tape:
      if self.variational:
        z_mean, z_log_var, z, rec = self(data, training=True)

        reconstruction_loss = tf.reduce_mean(tf.reduce_sum((data - rec)**2, axis=1))
        kl_loss = -0.5 * (1 + z_log_var - z_mean**2 - tf.exp(z_log_var))
        kl_loss = kl_weight*tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
      else:
        z, rec = self(data, training=True)
        reconstruction_loss = tf.reduce_mean(tf.reduce_sum((data - rec)**2, axis=1))
        kl_loss = 0

      total_loss = reconstruction_loss + kl_loss

    grads = tape.gradient(total_loss, self.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
    self.total_loss_tracker.update_state(total_loss)
    self.reconstruction_loss_tracker.update_state(reconstruction_loss)
    self.kl_loss_tracker.update_state(kl_loss/kl_weight)

    return {
        'loss': self.total_loss_tracker.result(),
        'mse': self.reconstruction_loss_tracker.result(),
        'kl': self.kl_loss_tracker.result(),
    }

  def call(self, inputs, training=False):
    if self.variational:
      z_mean, z_log_var, z = self.encoder(inputs, training=training)
      reconstruction = self.decoder(z, training=training)
      return z_mean, z_log_var, z, reconstruction
    else:
      z = self.encoder(inputs, training=training)
      reconstruction = self.decoder(z, training=training)
      return z, reconstruction


# class VariationalAutoEncoder(tf.keras.Model):
#   """Summary of class here"""

#   def __init__(self, **kwargs) -> None:
#     super().__init__()
#     default_kwargs = {
#       'latent_dim': 2,
#       'enc_layers': 4,
#       'enc_units': 256,
#       'enc_final_act': 'linear',
#       'enc_activation': 'leakyrelu',
#       'enc_leaky_rate': 0.2,
#       'enc_normalization': 'batch',
#       'enc_drop_rate': None,
#       'enc_loss_noise': 0.2,
#       'dec_layers': 4,
#       'dec_units': 256,
#       'dec_final_act': 'linear',
#       'dec_activation': 'leakyrelu',
#       'dec_leaky_rate': 0.2,
#       'dec_normalization': 'layer',
#       'dec_drop_rate': None,
#       'dec_loss_noise': 0.2,
#     }

#     self.info = default_kwargs | kwargs

# class AutoEncoder(tf.keras.Model):
#   """Summary of class here."""

#   def __init__(self, **kwargs) -> None:
#     super().__init__(**kwargs)
#     self.lay = tf.keras.Dense(100)

#   def call(self, inputs):
#     """Calls the model for inference."""
#     return inputs

#   def train_step(self, inputs):
#     """Train step."""
#     return inputs
