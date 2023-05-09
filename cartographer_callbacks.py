import sys
from typing import Callable

import transformers as tr
import datasets as ds
import numpy as np

class Cartographer(tr.TrainerCallback):
    def __init__(self, dataset: ds.Dataset,
                 trainer: tr.Trainer,
                 outputs_to_probabilities: Callable = None,
                 sparse_labels: bool = False,
                 ):
        """
        :param dataset: Dataset. Usually, as the paper suggests, this is the training dataset. It should be:
             - Non-shuffled, so each iteration over the dataset should yield samples in the same order
        :param outputs_to_probabilities: Callable to convert model's output to probabilities. Use this if the model
            outputs logits, dictionary or any other form which is not a vector of probabilities.
        :param sparse_labels: Set to ``True`` if the labels are given as integers (not one hot encoded)
        """

        self._dataset = dataset
        self._sparse_labels = sparse_labels

        # Stores the probabilities for the gold labels after each epoch,
        # e.g. self._gold_labels_probabilities[i] == gold label probabilities at the end of epoch i
        self._gold_labels_probabilities = []
        self.trainer = trainer
        self._outputs2probabilities = outputs_to_probabilities

    def on_epoch_end(self,  args: tr.TrainingArguments, state: tr.TrainerState, control: tr.TrainerControl, **kwargs):
        # Gather gold label probabilities over the dataset
        predictions = self.trainer.predict(self._dataset)
        probabilities = predictions[0]
        # predicted_labels_sparse = predictions[1]

        # Convert outputs to probabilities if necessary
        if self._outputs2probabilities is not None:
            probabilities = self._outputs2probabilities(probabilities)

        gold_probabilities = probabilities[np.arange(probabilities.shape[0]), self._dataset["label"]]
        gold_probabilities = np.expand_dims(gold_probabilities, axis=-1)
        self._gold_labels_probabilities.append(gold_probabilities)
        return

    @property
    def gold_labels_probabilities(self) -> np.ndarray:
        """
        Gold label predicted probabilities. With the shape of ``(n_samples, n_epochs)`` and ``(i, j)`` is the
        probability of the gold label for sample ``i`` at epoch ``j``
        :return: Gold label probabilities
        """
        return np.hstack(self._gold_labels_probabilities)

    @property
    def confidence(self) -> np.ndarray:
        """
        Average true label probability across epochs
        :return: Confidence
        """
        return np.mean(self.gold_labels_probabilities, axis=-1)

    @property
    def variability(self) -> np.ndarray:
        """
        Standard deviation of true label probability across epochs
        :return: Variability
        """

        return np.std(self.gold_labels_probabilities, axis=-1)

    @property
    def correctness(self) -> np.ndarray:
        """
        Proportion of correct predictions made across epochs.
        :return: Correctness
        """
        return np.mean(self.gold_labels_probabilities > 0.5, axis=-1)