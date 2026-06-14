#!/usr/bin/env python3
"""Defines a simple Generative Adversarial Network model."""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class Simple_GAN(keras.Model):
    """Simple Generative Adversarial Network model."""

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=0.005):
        """
        Initialize a Simple_GAN instance.

        Args:
            generator: Keras model that generates fake samples.
            discriminator: Keras model that classifies samples.
            latent_generator: Callable that generates latent vectors.
            real_examples: Tensor containing real training examples.
            batch_size: Number of samples used per training batch.
            disc_iter: Number of discriminator updates per train step.
            learning_rate: Learning rate for both Adam optimizers.
        """
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = 0.5
        self.beta_2 = 0.9

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
        Calculate the generator loss.

        Args:
            fake_output: Discriminator output for generated samples.

        Returns:
            Tensor containing the generator loss.
        """
        mse = tf.keras.losses.MeanSquaredError()
        return mse(fake_output, tf.ones(tf.shape(fake_output)))

    @staticmethod
    def discriminator_loss(real_output, fake_output):
        """
        Calculate the discriminator loss.

        Args:
            real_output: Discriminator output for real samples.
            fake_output: Discriminator output for generated samples.

        Returns:
            Tensor containing the discriminator loss.
        """
        mse = tf.keras.losses.MeanSquaredError()
        real_loss = mse(real_output, tf.ones(tf.shape(real_output)))
        fake_loss = mse(fake_output, -tf.ones(tf.shape(fake_output)))
        return real_loss + fake_loss

    def get_fake_sample(self, size=None, training=False):
        """
        Generate fake samples using the generator.

        Args:
            size: Number of fake samples to generate.
            training: Whether the generator should run in training mode.

        Returns:
            Tensor containing generated samples.
        """
        if size is None:
            size = self.batch_size
        latent_sample = self.latent_generator(size)
        return self.generator(latent_sample, training=training)

    def get_real_sample(self, size=None):
        """
        Select a random batch of real samples.

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

    def train_step(self, data):
        """
        Perform one GAN training step.

        Args:
            data: Unused argument required by the Keras train_step API.

        Returns:
            Dictionary containing discriminator and generator losses.
        """
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                real_sample = self.get_real_sample()
                fake_sample = tf.stop_gradient(
                    self.get_fake_sample(training=False)
                )

                real_output = self.discriminator(real_sample, training=True)
                fake_output = self.discriminator(fake_sample, training=True)
                discr_loss = self.discriminator.loss(real_output, fake_output)

            discr_gradients = tape.gradient(
                discr_loss,
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
            "gen_loss": gen_loss
        }
