from cartographer_callbacks import Cartographer, NormalizedCartographer
import transformers as tr
import datasets as ds
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt
import seaborn as sns


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
        # TODO Remove sharding when not just testing anymore
        tokenized_dataset[key] = tokenized_dataset[key].shard(4000,1)

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
        softmax_scores = softmax(predictions, axis=1)
        return softmax_scores

    cartographer_global_norm1 = NormalizedCartographer(tokenized_dataset['train'],
                                trainer=trainer,
                                outputs_to_probabilities=calc_probs,
                                mode='global'
                                )

    trainer.add_callback(cartographer_global_norm1)

    # Fine-tune the model
    print("start training")
    trainer.train()

    cartographer_global_norm2 = NormalizedCartographer(tokenized_dataset['train'],
                                                      trainer=trainer,
                                                      outputs_to_probabilities=calc_probs,
                                                      mode='global'
                                                      )
    trainer = tr.Trainer(
        model=trainer.model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
    )
    trainer.add_callback(cartographer_global_norm2)
    trainer.train()

    cartographer_global_norm1.compare_to(cartographer_global_norm2)

    # If everything works during representation it should simply invert the vectors
    # cartographer_global_norm2.compare_to(cartographer_global_norm1)

    # Saves used Dataset with learnability, variability and confidence of each sample
    cartographer_global_norm1.save_map("test")


    print(cartographer_global_norm1.learnability)
