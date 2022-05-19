"""Module holding classes for implementing (Varitational) AutoEncoders."""
from typing import Union, Callable

import numpy as np
import tensorflow as tf


class VariationalAutoEncoder(tf.keras.Model):
  """Summary of class here"""

  def __init__(self, **kwargs) -> None:
    super().__init__()
    default_kwargs = {
      'latent_dim': 2,
      'enc_layers': 4,
      'enc_units': 256,
      'enc_final_act': 'linear',
      'enc_activation': 'leakyrelu',
      'enc_leaky_rate': 0.2,
      'enc_normalization': 'batch',
      'enc_drop_rate': None,
      'enc_loss_noise': 0.2,
      'dec_layers': 4,
      'dec_units': 256,
      'dec_final_act': 'linear',
      'dec_activation': 'leakyrelu',
      'dec_leaky_rate': 0.2,
      'dec_normalization': 'layer',
      'dec_drop_rate': None,
      'dec_loss_noise': 0.2,
    }

    self.info = default_kwargs | kwargs


class AutoEncoder(tf.keras.Model):
  """Summary of class here."""

  def __init__(self) -> None:
    super().__init__()
    self.lay = tf.keras.layers.Dense(100)

  def call(self, inputs):
    """Calls the model for inference."""
    return inputs

  def train_step(self, inputs):
    """Train step."""
    return inputs
