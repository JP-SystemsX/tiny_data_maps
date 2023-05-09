from cartographer_callbacks import Cartographer
import transformers as tr
import datasets as ds
import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt
import seaborn as sns

def plot_map(cartographer: Cartographer):
    # Plot
    _, ax = plt.subplots(figsize=(9, 7))

    sns.scatterplot(x=cartographer.variability, y=cartographer.confidence, hue=cartographer.correctness,
                    ax=ax)
    sns.kdeplot(x=cartographer.variability, y=cartographer.confidence,
                levels=8, color=sns.color_palette("Paired")[7], linewidths=1, ax=ax)

    ax.set(title='Data map for QNLI train set\nbased on a DistilBERT classifier',
           xlabel='Variability', ylabel='Confidence')

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

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Load the dataset
    dataset = ds.load_dataset("imdb")

    # Load the tokenizer
    tokenizer = tr.AlbertTokenizerFast.from_pretrained("albert-base-v2")

    # Define the model
    model = tr.AlbertForSequenceClassification.from_pretrained("albert-base-v2", num_labels=2)


    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)


    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    for key in tokenized_dataset:
        tokenized_dataset[key] = tokenized_dataset[key].shard(1000,1)

    # Define the training arguments
    training_args = tr.TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=500,
    )

    # Define the trainer
    trainer = tr.Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
    )

    def calc_probs(predictions):
        sigmoid_scores = expit(predictions)
        return sigmoid_scores / np.sum(sigmoid_scores, axis=1, keepdims=True)

    cartographer = Cartographer(tokenized_dataset['train'],
                                sparse_labels=True,
                                trainer=trainer,
                                outputs_to_probabilities=calc_probs)
    trainer.add_callback(cartographer)

    # Fine-tune the model
    print("start training")
    trainer.train()
    plot_map(cartographer)
