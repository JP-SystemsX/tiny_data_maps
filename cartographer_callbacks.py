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
                 #sparse_labels: bool = False,
                 ):
        """
        :param dataset: Dataset. Usually, as the paper suggests, this is the training dataset. It should be:
             - Non-shuffled, so each iteration over the dataset should yield samples in the same order
             - Labels should be sparse encoded i.e. as integer not one-hot
             - label-column shall be called 'label' and only contain a single label per row
        :param outputs_to_probabilities: Callable to convert model's output to probabilities. Use this if the model
            outputs logits, dictionary or any other form which is not a vector of probabilities.
        """
        #:param sparse_labels: Set to ``True`` if the labels are given as integers (not one hot encoded)

        self.dataset = dataset
        #self.sparse_labels = sparse_labels

        # Stores the probabilities for the gold labels after each epoch,
        # e.g. self._gold_labels_probabilities[i] == gold label probabilities at the end of epoch i
        self._gold_labels_probabilities = []
        self.trainer = trainer
        self.outputs2probabilities = outputs_to_probabilities

    def on_epoch_end(self,  args: tr.TrainingArguments, state: tr.TrainerState, control: tr.TrainerControl, **kwargs):
        # Gather gold label probabilities over the dataset
        predictions = self.trainer.predict(self.dataset)
        probabilities = predictions[0]
        # predicted_labels_sparse = predictions[1]

        # Convert outputs to probabilities if necessary
        if self.outputs2probabilities is not None:
            probabilities = self.outputs2probabilities(probabilities)

        gold_probabilities = probabilities[np.arange(probabilities.shape[0]), self.dataset["label"]]
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
            title=f'Data map for {self.dataset.builder_name} set\nbased on a {self.trainer.model.name_or_path} classifier',
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


class NormalizedCartographer(Cartographer):
    """
    A Cartographer that is normalized such that easy to learn samples have a large positive y val, hard to learn
    a large negative one while ambiguous samples have a low x value.
    The intention of this is to use x*y as a 1D-heuristic of how easy a sample is to learn.
    (i.e. the higher the easier)
    """

    def __init__(self, dataset: ds.Dataset,
                 trainer: tr.Trainer,
                 outputs_to_probabilities: Callable = None,
                 # sparse_labels: bool = False,
                 mode='global'
                 ):
        super().__init__(dataset=dataset,trainer=trainer,outputs_to_probabilities=outputs_to_probabilities)
        self.mode = mode

    @property
    def confidence(self) -> np.ndarray:
        """
        Average true label probability across epochs normalized either by considering
        the true absolute maximum and minimum that can be possible (1,0) (i.e. 'global' mode)
        or by considering the real max and minimum that was achieved 0 <= min <= max <= 1 (i.e. 'local' mode)
        :return: Confidence
        """
        if self.mode == 'local':
            mean = np.mean(self.gold_labels_probabilities, axis=-1)
            mean_norm = (mean - np.min(mean))
            mean_norm = mean_norm / np.max(mean_norm)
            return mean_norm*2 -1
        elif self.mode == 'global':
            return np.mean(self.gold_labels_probabilities, axis=-1)*2 - 1

    @property
    def solidity(self) -> np.ndarray:
        """
        1 - Standard deviation of true label probability across epochs.
        Such that larger values mean more extreme (i.e. doesn't change at all)
        compared to variability where smaller values mean more extreme.
        :return: Solidity
        """
        return 1 - np.std(self.gold_labels_probabilities, axis=-1)

    @property
    def learnability(self):
        '''
        A simple heuristic on how learnable a sample is.
        Calculated simply by multiplying solidity and (normalized) confidence
        leading to a value between [+1,-1] where:
        +1 means super-easy-to-learn (Model might have over-fitted on that one)
         0 means ambiguous
        -1 means couldn't be learned at all (Might be falsely Labeled)
        :return: Learnabilty
        '''
        return self.solidity * self.confidence

    def visualize(self):
        # Plot
        _, ax = plt.subplots(figsize=(9, 7))

        sns.scatterplot(x=self.solidity, y=self.confidence, hue=self.correctness,
                        ax=ax)
        sns.kdeplot(x=self.solidity, y=self.confidence,
                    levels=8, color=sns.color_palette("Paired")[7], linewidths=1, ax=ax)

        ax.set(
            title=f'Data map for {self.dataset.builder_name} set\nbased on a {self.trainer.model.name_or_path} classifier',
            xlabel='Solidity',
            ylabel='Confidence'
        )

        # Annotations
        box_style = {'boxstyle': 'round', 'facecolor': 'white', 'ec': 'black'}
        ax.text(0.8, 0.85,
                'easy-to-learn',
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=box_style)
        ax.text(0.05, 0.51,
                'ambiguous',
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=box_style)
        ax.text(0.8, 0.1,
                'hard-to-learn',
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=box_style)

        ax.legend(title='Correctness')

        # Set the y-axis limits to be between 1 and -1
        plt.ylim(-1.1, 1.1)
        plt.xlim(np.min(self.solidity), 1)

        plt.axhline(y=0, color='black', linestyle='--')

        plt.show()

    def compare_to(self, other:Cartographer):
        positions_x = self.solidity
        positions_y = self.confidence
        movements_x = other.solidity - self.solidity
        movements_y = other.confidence - self.confidence

        # Plot
        _, ax = plt.subplots(figsize=(9, 7))

        sns.scatterplot(x=positions_x, y=positions_y, hue=self.correctness,
                        ax=ax)
        sns.kdeplot(x=positions_x, y=positions_y,
                    levels=8, color=sns.color_palette("Paired")[7], linewidths=1, ax=ax)

        ax.set(
            title=f'''
            Data map comparison for:
            {self.dataset.builder_name} set\nbased on a {self.trainer.model.name_or_path} classifier and
            {other.dataset.builder_name} set\nbased on a {other.trainer.model.name_or_path} classifier 
            ''',
            xlabel='Solidity',
            ylabel='Confidence'
        )

        # Annotations
        box_style = {'boxstyle': 'round', 'facecolor': 'white', 'ec': 'black'}
        ax.text(0.8, 0.85,
                'easy-to-learn',
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=box_style)
        ax.text(0.05, 0.51,
                'ambiguous',
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=box_style)
        ax.text(0.8, 0.1,
                'hard-to-learn',
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=box_style)

        ax.legend(title='Correctness')

        # Set the y-axis limits to be between 1 and -1
        plt.ylim(-1.1, 1.1)
        plt.xlim(min(np.min(self.solidity), np.min(other.solidity)), 1)

        plt.axhline(y=0, color='black', linestyle='--')

        plt.quiver(positions_x, positions_y, movements_x, movements_y, angles='xy', scale_units='xy', scale=1, linewidth=0.01)

        plt.show()


