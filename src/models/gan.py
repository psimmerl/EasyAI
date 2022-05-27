"""Module holding classes for implementing Generative Adversarial Networks."""
from typing import Union, Callable

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.layers import (Input, Dense, Dropout, BatchNormalization,
                                     LayerNormalization, LeakyReLU)
from tensorflow.keras.regularizers import L2
from .losses import WassersteinLoss, GradientPenalty


class GenerativeAdversarialNetwork(tf.keras.Model):
  """Summary of class here."""

  def __init__(self, n_feats, **kwargs) -> None:
    super().__init__(**kwargs)
    default_kwargs = {
        'latent_dim': 128,
        'gen_layers': 4,
        'gen_units': 256,
        'gen_final_act': 'linear',
        'gen_activation': 'leakyrelu',
        'gen_leaky_rate': 0.2,
        'gen_normalization': 'batch',
        'gen_drop_rate': None,
        'gen_loss_noise': 0.2,
        'dis_layers': 4,
        'dis_units': 256,
        'dis_final_act': 'linear',
        'dis_activation': 'leakyrelu',
        'dis_leaky_rate': 0.2,
        'dis_normalization': 'layer',
        'dis_drop_rate': None,
        'dis_loss_noise': 0.2,
        'dis_extra_steps': 5
    }
    kwargs = default_kwargs | kwargs
    self.info = kwargs
    self.info['n_feats'] = n_feats

    self.generator = self.build_submodel('generator', **self.info)
    self.discriminator = self.build_submodel('discriminator', **self.info)

  def build_submodel(self, model: str, **kwargs) -> tf.keras.Model:
    """Builds the GAN sub models."""
    settings = {
        'name': 'Generator',
        'input': kwargs['latent_dim'],
        'output': kwargs['n_feats']
    }
    wt_decay = None
    if model in ('d', 'dis', 'discriminator'):
      settings['name'] = 'Discriminator'
      settings['input'] = kwargs['n_feats']
      settings['output'] = 1
      if kwargs['dis_normalization']:
        wt_decay = 1e-3

    for k, v in kwargs.items():
      if k[:3] == settings['name'][:3].lower():
        if isinstance(v, str):
          v = v.lower()
        settings[k[4:]] = v  # skip first four: 'gen_' or 'dis_'

    activation = settings['activation'].lower()

    input_ = Input(shape=(settings['input'], ))
    for ilay in range(settings['layers']):
      if activation == 'leakyrelu':
        x = Dense(units=settings['units'],
                  activation='linear',
                  kernel_initializer='he_normal',
                  kernel_regularizer=(L2(wt_decay) if wt_decay else None),
                  )(x if ilay else input_)
        x = LeakyReLU(alpha=settings['leaky_rate'])(x)
      else:
        x = Dense(units=settings['units'],
                  activation=activation,
                  kernel_initializer='he_normal',
                  kernel_regularizer=(L2(wt_decay) if wt_decay else None),
                  )(x if ilay else input_)

      if settings['drop_rate']:
        x = Dropout(settings['drop_rate'])(x)
      if settings['normalization'] in ('b', 'batch'):
        x = BatchNormalization()(x)
      elif settings['normalization'] in ('l', 'layer'):
        x = LayerNormalization()(x)
    output_ = Dense(units=settings['output'], activation=settings['final_act'])(x)

    return tf.keras.Model(input_, output_, name=settings['name'])

  def compile(self,
              g_opt: Union[tf.keras.optimizers.Optimizer, str] = 'adam',
              d_opt: Union[tf.keras.optimizers.Optimizer, str] = 'adam',
              loss: Union[Callable, tf.losses.Loss] = WassersteinLoss(),
              **kwargs) -> None:
    """Compiles model."""

    default_kwargs = {
        'gradient_penalty_weight': 10,
        'gradient_penalty_form': 'normal',
    }
    kwargs = default_kwargs | kwargs

    super().compile(**kwargs)

    self.penalty = None
    if isinstance(loss, WassersteinLoss) and kwargs['gradient_penalty_weight']:
      self.penalty = GradientPenalty(weight=kwargs['gradient_penalty_weight'],
                                     form=kwargs['gradient_penalty_form'])
    else:
      kwargs['gradient_penalty_weight'] = None

    default_opts = {
        'gen_adam': Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9),
        'dis_adam': Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
    }
    if isinstance(g_opt, str):
      g_opt = default_opts['gen_' + g_opt]
    if isinstance(d_opt, str):
      d_opt = default_opts['dis_' + d_opt]

    self.gen_opt, self.dis_opt, self.loss = g_opt, d_opt, loss
    self.info = self.info | kwargs

  def call(self, inputs, model='generator', lvs=False, training=False):
    """Calls the model for inference."""
    #TODO: Make this run in batches to avoid memory overflow
    if model in ('g', 'generator'):
      if not lvs:
        inputs = tf.random.normal(shape=(inputs, self.info['latent_dim']))
      return self.generator(inputs, training=training)
    return self.discriminator(inputs, training=training)

  def train_step(self, real):
    """Train step."""
    batch_size = tf.shape(real)[0]

    for istep in range(self.info['dis_extra_steps']):
      rlv = tf.random.normal(shape=(batch_size, self.info['latent_dim']))
      fake = self.generator(rlv, training=False)
      with tf.GradientTape() as tape:
        real_logits = self.discriminator(real, training=True)
        fake_logits = self.discriminator(fake, training=True)
        if self.info['dis_loss_noise']:
          real_logits *= tf.random.normal((batch_size, 1), 1,
                                          self.info['dis_loss_noise'])
          fake_logits *= tf.random.normal((batch_size, 1), 1,
                                          self.info['dis_loss_noise'])
        dis_loss = self.loss(real_logits, fake_logits)

        dis_cost = dis_loss
        if self.info['gradient_penalty_weight']:
          dis_cost += self.penalty.call(real, fake, self.discriminator)

      dis_grad = tape.gradient(dis_cost,
                               self.discriminator.trainable_variables)

      self.dis_opt.apply_gradients(
          zip(dis_grad, self.discriminator.trainable_variables))

    rlv = tf.random.normal(shape=(batch_size, self.info['latent_dim']))
    with tf.GradientTape() as tape:
      gen = self.generator(rlv, training=True)
      gen_logits = self.discriminator(gen, training=False)
      if self.info['gen_loss_noise']:
        gen_logits *= tf.random.normal((batch_size, 1), 1,
                                       self.info['gen_loss_noise'])
      gen_loss = self.loss(gen_logits, tf.zeros_like(gen_logits))

    gen_grad = tape.gradient(gen_loss, self.generator.trainable_variables)
    self.gen_opt.apply_gradients(
        zip(gen_grad, self.generator.trainable_variables))

    return {'gloss': gen_loss, 'dloss': dis_loss}

  def get_config(self):
    return self.info

  @classmethod
  def from_config(cls, config):
    return cls(**config)


class GANMonitor(tf.keras.callbacks.Callback):
  """GAN Monitor"""

  def __init__(self,
              training_data,
              validation_data,
              batch_size=1024,
              #  log_dir: str='logs/',
              freq: int=1,
              **kwargs) -> None:
    super().__init__(**kwargs)
    # self.log_dir = log_dir.rstrip('/')
    self.training_data = training_data
    self.validation_data = validation_data
    self.batch_size = batch_size
    self.freq = freq

  def on_train_begin(self, logs=None):
    self.trn_rmse = []
    self.val_rmse = []

  def on_epoch_end(self, epoch, logs=None) -> None:
    if epoch%self.freq != 0:
      return

    epoch+=1
    # self.model.save(f'{self.log_dir}/GAN_epoch{epoch}', save_format='tf')

    for real in (self.training_data, self.validation_data):
      for batch_idx in range(0, len(real), self.batch_size):
        count = min(len(real)-batch_idx, self.batch_size)
        temp = np.asarray(self.model(count, model='g', training=False))

        if batch_idx == 0:
          fake = temp
        else:
          fake = np.vstack((fake, temp))

      rmse = np.sqrt(np.mean(np.sum((real-fake),axis=0)**2))
      if real is self.training_data:
        self.trn_rmse.append(rmse)
        logs['trn_rmse'] = rmse
        # print(f'- trn_rmse: {rmse:.0f}', end='')
      else:
        self.val_rmse.append(rmse)
        logs['val_rmse'] = rmse
        # print(f'- val_rmse: {rmse:.0f}', end='')
    # print()
