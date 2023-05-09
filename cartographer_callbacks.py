import sys
from typing import Callable

import transformers as tr
import datasets as ds
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

    def visualize(self):
        # Plot
        _, ax = plt.subplots(figsize=(9, 7))

        sns.scatterplot(x=self.variability, y=self.confidence, hue=self.correctness,
                        ax=ax)
        sns.kdeplot(x=self.variability, y=self.confidence,
                    levels=8, color=sns.color_palette("Paired")[7], linewidths=1, ax=ax)

        ax.set(
            title=f'Data map for {self._dataset.builder_name} set\nbased on a {self.trainer.model.name_or_path} classifier',
            xlabel='Variability',
            ylabel='Confidence'
        )

        # Annotations
        box_style = {'boxstyle': 'round', 'facecolor': 'white', 'ec': 'black'}
        ax.text(0.14, 0.84,
                'easy-to-learn',
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=box_style)
        ax.text(0.75, 0.5,
                'ambiguous',
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=box_style)
        ax.text(0.14, 0.14,
                'hard-to-learn',
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=box_style)

        ax.legend(title='Correctness')
        plt.show()


class Normalized_Cartographer(Cartographer):
    """
    A Cartographer that is normalized such that easy to learn samples have a large positive y val, hard to learn
    a large negative one while ambiguous samples have a low x value.
    The intention of this is to use x*y as a 1D-heuristic of how easy a sample is to learn.
    (i.e. the higher the easier)
    """