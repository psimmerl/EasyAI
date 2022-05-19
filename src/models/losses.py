"""Module holding custom loss functions."""
import numpy as np
import tensorflow as tf


class WassersteinLoss(tf.losses.Loss):
  """Wasserstein Loss"""

  def __init__(self, name: str = 'wasserstein') -> None:
    super().__init__(name=name)

  def call(self, real_logits, fake_logits) -> tf.Tensor:
    """call"""
    return tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)


class GradientPenalty(tf.losses.Loss):
  """Gradient Penalty for Wasserstein Loss. (GP is a regularizer)"""

  def __init__(self,
               name: str = 'gradient_penalty',
               weight: float = 10,
               form: str = 'normal') -> None:
    super().__init__(name=name)
    self.weight = weight

  def call(self, real: tf.Tensor, fake: tf.Tensor,
           discriminator: tf.keras.Model) -> tf.Tensor:
    """Gradient Penalty call"""
    batch_size = tf.shape(real)[0]
    alpha = tf.random.normal(shape=(batch_size, 1))
    interpolated = real + alpha * (fake - real)

    with tf.GradientTape() as tape:
      tape.watch(interpolated)
      pred = discriminator(interpolated, training=True)
    grads = tape.gradient(pred, [interpolated])[0]

    norm = tf.sqrt(tf.reduce_sum(grads**2, axis=1))
    penalty = tf.reduce_mean((norm - 1)**2)

    return self.weight * penalty


class BCELoss(tf.losses.Loss):
  """BCE Loss"""

  def __init__(self, name='bce_gan') -> None:
    super().__init__(name=name)


class MMDLoss(tf.losses.Loss):
  """MMD Loss"""

  def __init__(self, name='mmd') -> None:
    super().__init__(name=name)


class RaLSLoss(tf.losses.Loss):
  """RaLS Loss"""

  def __init__(self, name='rals') -> None:
    super().__init__(name=name)


if __name__ == '__main__':
  np.random.seed(42)
  loss = WassersteinLoss()
  print(tf.__version__)
