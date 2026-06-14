#!/usr/bin/env python3
"""Defines a Wasserstein GAN model with gradient penalty."""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class WGAN_GP(keras.Model):
    """Wasserstein GAN model using gradient penalty."""

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=0.005, lambda_gp=10):
        """
        Initialize a WGAN_GP instance.

        Args:
            generator: Keras model used to generate fake samples.
            discriminator: Keras model used as the Wasserstein critic.
            latent_generator: Callable that generates latent vectors.
            real_examples: Tensor containing real training examples.
            batch_size: Number of samples used in each training batch.
            disc_iter: Number of discriminator updates per train step.
            learning_rate: Learning rate for the Adam optimizers.
            lambda_gp: Weight of the gradient penalty term.
        """
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = 0.3
        self.beta_2 = 0.9

        self.lambda_gp = lambda_gp
        self.dims = self.real_examples.shape
        self.len_dims = len(self.dims)
        self.axis = tf.range(1, self.len_dims, delta=1, dtype="int32")

        self.scal_shape = self.dims.as_list()
        self.scal_shape[0] = self.batch_size
        for i in range(1, self.len_dims):
            self.scal_shape[i] = 1
        self.scal_shape = tf.convert_to_tensor(self.scal_shape)

        self.generator.loss = self.generator_loss
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2
        )
        self.generator.compile(
            optimizer=self.generator.optimizer,
            loss=self.generator.loss
        )

        self.discriminator.loss = self.discriminator_loss
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2
        )
        self.discriminator.compile(
            optimizer=self.discriminator.optimizer,
            loss=self.discriminator.loss
        )

    @staticmethod
    def generator_loss(fake_output):
        """
        Calculate the Wasserstein generator loss.

        Args:
            fake_output: Discriminator output for fake samples.

        Returns:
            Tensor containing the generator loss.
        """
        return -tf.reduce_mean(fake_output)

    @staticmethod
    def discriminator_loss(real_output, fake_output):
        """
        Calculate the Wasserstein discriminator loss.

        Args:
            real_output: Discriminator output for real samples.
            fake_output: Discriminator output for fake samples.

        Returns:
            Tensor containing the discriminator loss.
        """
        real_loss = tf.reduce_mean(real_output)
        fake_loss = tf.reduce_mean(fake_output)
        return fake_loss - real_loss

    def get_fake_sample(self, size=None, training=False):
        """
        Generate fake samples.

        Args:
            size: Number of fake samples to generate.
            training: Whether the generator runs in training mode.

        Returns:
            Tensor containing generated fake samples.
        """
        if size is None:
            size = self.batch_size
        latent_sample = self.latent_generator(size)
        return self.generator(latent_sample, training=training)

    def get_real_sample(self, size=None):
        """
        Select random real samples.

        Args:
            size: Number of real samples to select.

        Returns:
            Tensor containing randomly selected real samples.
        """
        if size is None:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def get_interpolated_sample(self, real_sample, fake_sample):
        """
        Generate interpolated samples between real and fake samples.

        Args:
            real_sample: Tensor containing real samples.
            fake_sample: Tensor containing fake samples.

        Returns:
            Tensor containing interpolated samples.
        """
        alpha = tf.random.uniform(self.scal_shape)
        beta = tf.ones(self.scal_shape) - alpha
        return alpha * real_sample + beta * fake_sample

    def gradient_penalty(self, interpolated_sample):
        """
        Compute the gradient penalty for interpolated samples.

        Args:
            interpolated_sample: Tensor containing interpolated samples.

        Returns:
            Tensor containing the gradient penalty.
        """
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_sample)
            pred = self.discriminator(interpolated_sample, training=True)

        grads = gp_tape.gradient(pred, [interpolated_sample])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=self.axis))
        return tf.reduce_mean((norm - 1.0) ** 2)

    def train_step(self, data):
        """
        Perform one WGAN-GP training step.

        Args:
            data: Unused argument required by the Keras train_step API.

        Returns:
            Dictionary containing discriminator loss, generator loss,
            and gradient penalty.
        """
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                real_sample = self.get_real_sample()
                fake_sample = tf.stop_gradient(
                    self.get_fake_sample(training=False)
                )
                interpolated_sample = self.get_interpolated_sample(
                    real_sample,
                    fake_sample
                )

                real_output = self.discriminator(real_sample, training=True)
                fake_output = self.discriminator(fake_sample, training=True)

                discr_loss = self.discriminator.loss(
                    real_output,
                    fake_output
                )
                gp = self.gradient_penalty(interpolated_sample)
                new_discr_loss = discr_loss + self.lambda_gp * gp

            discr_gradients = tape.gradient(
                new_discr_loss,
                self.discriminator.trainable_variables
            )
            self.discriminator.optimizer.apply_gradients(
                zip(discr_gradients, self.discriminator.trainable_variables)
            )

        with tf.GradientTape() as tape:
            fake_sample = self.get_fake_sample(training=True)
            fake_output = self.discriminator(fake_sample, training=False)
            gen_loss = self.generator.loss(fake_output)

        gen_gradients = tape.gradient(
            gen_loss,
            self.generator.trainable_variables
        )
        self.generator.optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables)
        )

        return {
            "discr_loss": discr_loss,
            "gen_loss": gen_loss,
            "gp": gp
        }
