import tensorflow as tf
import keras
from keras.models import Model

class PRMCModel(Model):
    """
    PRMCModel for keras. Special train functions etc
    """
    def compile(self, lossf: keras.losses.Loss, *args, **kwargs):
        """
        Compile the model
        Args:
            lossf (keras.losses.Loss): loss_function of Loss-type
        """
        self.loss_PRMC = lossf
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.val_loss_tracker = keras.metrics.Mean(name='val_loss')
        super().compile(*args, **kwargs)

    def train(self, train_dataset: tf.data.Dataset, lossf: keras.losses.Loss, val_dataset: tf.data.Dataset=None, epochs: int=1, callbacks=[]):
        """
        train the model

        Args:
            train_dataset (tf.data.Dataset): dataset to train on
            lossf (keras.losses.Loss): lossfunction to use
            val_dataset (tf.data.Dataset, optional): dataset to validate on. Defaults to None.
            epochs (int, optional): num of epochs to train. Defaults to 1.
            callbacks (list, optional): callbacks to be passed on. Defaults to [].
        """
        if len(train_dataset.element_spec) == 3:  # means that individual ruls are passed
            self._train_with_sample_weight(
                train_dataset, lossf, val_dataset=val_dataset, epochs=epochs, callbacks=callbacks)
        else:
            self._train(train_dataset, lossf, val_dataset=val_dataset,
                        epochs=epochs, callbacks=callbacks)

    def _train(self, train_dataset: tf.data.Dataset, lossf: keras.losses.Loss, val_dataset: tf.data.Dataset=None, epochs: int=1, callbacks=[]):
        """
        Train the model, with shared cost of RUL
        Used for training a model with individual costs of rul
        Args:
            train_dataset (tf.data.Dataset): dataset to train on
            lossf (keras.losses.Loss): lossfunction to use
            val_dataset (tf.data.Dataset, optional): dataset to validate on. Defaults to None.
            epochs (int, optional): num of epochs to train. Defaults to 1.
            callbacks (list, optional): callbacks to be passed on. Defaults to [].
        """
        print(
            f"Training with one shared cost of rul: {self.loss_PRMC.shared_cost_rul}")
        for epoch in range(epochs):
            print(f"\nStart of epoch {epoch}")

            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

                with tf.GradientTape() as tape:
                    preds = self(x_batch_train, training=True)
                    loss_value = lossf(y_batch_train, preds)

                grads = tape.gradient(loss_value, self.trainable_weights)
                self.optimizer.apply_gradients(
                    zip(grads, self.trainable_weights))
                self.loss_tracker.update_state(loss_value)
            if val_dataset is not None:
                for x_batch_val, y_batch_val in val_dataset:
                    preds_val = self(x_batch_val, training=False)
                    val_loss = lossf(y_batch_val, preds_val)
                    self.val_loss_tracker.update_state(val_loss)

            print(
                f"loss: {self.loss_tracker.result()} val_loss: {self.val_loss_tracker.result()}")

    def _train_with_sample_weight(self, train_dataset: tf.data.Dataset, lossf: keras.losses.Loss, val_dataset: tf.data.Dataset=None, epochs: int=1, callbacks=[]):
        """
        Used for training a model with individual costs of rul
        Args:
            train_dataset (tf.data.Dataset): dataset to train on
            lossf (keras.losses.Loss): lossfunction to use
            val_dataset (tf.data.Dataset, optional): dataset to validate on. Defaults to None.
            epochs (int, optional): num of epochs to train. Defaults to 1.
            callbacks (list, optional): callbacks to be passed on. Defaults to [].
        """
        print(
            f"Training with individual costs of rul; dataset dim: {train_dataset.element_spec}")
        for epoch in range(epochs):
            print(f"\nStart of epoch {epoch}")

            for step, (x_batch_train, y_batch_train, ul_batch_train) in enumerate(train_dataset):

                with tf.GradientTape() as tape:
                    preds = self(x_batch_train, training=True)
                    loss_value = lossf(y_batch_train, preds,
                                       ind_cost_rul=ul_batch_train)

                grads = tape.gradient(loss_value, self.trainable_weights)
                self.optimizer.apply_gradients(
                    zip(grads, self.trainable_weights))
                self.loss_tracker.update_state(loss_value)

            if val_dataset is not None:
                for x_batch_val, y_batch_val, ul_batch_val in val_dataset:
                    preds_val = self(x_batch_val, training=False)
                    val_loss = lossf(y_batch_val, preds_val,
                                     ind_cost_rul=ul_batch_val)
                    self.val_loss_tracker.update_state(val_loss)

            print(
                f"loss: {self.loss_tracker.result()} val_loss: {self.val_loss_tracker.result()}")
