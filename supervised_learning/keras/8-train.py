#!/usr/bin/env python3
"""Something that function does"""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None,
                early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1,
                decay_rate=1, save_best=False,
                filepath=None,
                verbose=True, shuffle=False):
    """Something that function does"""

    if not validation_data:

        history = network.fit(
            x=data,
            y=labels,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            shuffle=shuffle
        )
    else:
        callbacks = []
        if early_stopping:
            earlystop = K.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience
            )
            callbacks.append(earlystop)

        if learning_rate_decay:
            def lr_decay(epoch):
                lr = alpha / (1 + decay_rate * epoch)
                return lr

            lr_scheduler = K.callbacks.LearningRateScheduler(
                lr_decay, verbose=1
            )
            callbacks.append(lr_scheduler)
        checkpoint_cb = K.callbacks.ModelCheckpoint(
            filepath,
            save_best_only=filepath)
        callbacks.append(checkpoint_cb)
        history = network.fit(
            x=data,
            y=labels,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            shuffle=shuffle,
            callbacks=callbacks,
            validation_data=validation_data
        )

    return history
