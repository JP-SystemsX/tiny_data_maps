import warnings

from cartographer_callbacks import NormalizedCartographer
import transformers as tr
import datasets as ds
from scipy.special import softmax
import uuid
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='''
    A simple script to download a dataset from huggingface 
    and save it a specified address for further usage (e.g. preprocessing).
    ''')

    parser.add_argument('--dataset_address', type=str, required=True,
                        help='Address of the Dataset')

    parser.add_argument('--output_address', type=str, required=True,
                        help='Path to directory where the final Dataset shall be saved')

    parser.add_argument('--cache_adr', type=str,
                        help='Path to directory where all the downloaded stuff shall be cached')

    parser.add_argument('--model_name', type=str, default="albert-base-v2",
                        help='Huggingface model name')

    parser.add_argument("--repetitions", type=int, default=1,
                        help="""
                        How often shall process be repeated to get more stable results. 
                        (Note: DS-Maps vary tremendously between runs)
                        """
                        )

    parser.add_argument("--epochs", type=int, default=10,
                        help="""
                            For how many epochs per repetition shall it be trained?
                            High numbers means training till death 
                            --> useful for finding Too hard to learn samples 
                                i.e. samples that still don't have high confidence (around >90%) 
                                might be misclassified or the labels don't do it justice like a zebra 
                                in a cat-dog dataset
                            Low/Medium numbers means true DS-Maps
                            Note: what is a high and a low number depends on the DS and model used
                            """
                        )

    parser.add_argument("--batch_size", type=int, default=8,
                        help="""
                            How often shall process be repeated to get more stable results. 
                            (Note: DS-Maps vary tremendously between runs)
                            """
                        )

    return parser.parse_args()

# Simple variable to avoid resetting cartographer over epochs that would be a waste
cartographer_failsave = 0
def make_trainer(MODEL_NAME, training_args, tokenized_dataset, cartographer=None, num_labels=2):
    # Define the model
    model = tr.AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels, cache_dir=args.cache_adr)
    # Define the trainer
    trainer = tr.Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
    )

    if cartographer is None:
        warnings.warn("No Cartographer Given in first repetition expected but else problem")
        global cartographer_failsave
        assert cartographer_failsave == 0
        cartographer_failsave += 1
        cartographer = NormalizedCartographer(
            tokenized_dataset,
            trainer=trainer,
            outputs_to_probabilities=calc_probs,
            mode='global'
        )
    else:
        cartographer.trainer = trainer

    trainer.add_callback(cartographer)
    return trainer, cartographer

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args = parse_args()
    MODEL_NAME = args.model_name

    # Load the dataset
    #dataset = ds.load_from_disk(args.dataset_address)
    dataset = ds.load_dataset(args.dataset_address, cache_dir=args.cache_adr)
    if "validation" in dataset.keys():  # Complicated way of saying drop validation set
        dataset = ds.DatasetDict({
            # "train": ds.concatenate_datasets([dataset["train"], dataset["validation"]]),
            "train": dataset["train"],
            "test": dataset["test"]
        })
    label_set = set(dataset["train"]["label"])

    # Load the tokenizer
    tokenizer = tr.AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=args.cache_adr)

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)


    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    """for key in tokenized_dataset:
        # TODO Remove sharding when not just testing anymore
        tokenized_dataset[key] = tokenized_dataset[key].shard(4000,1)"""

    # Define the training arguments
    training_args = tr.TrainingArguments(
        output_dir='./results',
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=500,
    )

    def calc_probs(predictions):
        softmax_scores = softmax(predictions, axis=1)
        return softmax_scores



    # Fine-tune the model
    print("start training")
    cartographer = None
    for r in range(args.repetitions):
        trainer, cartographer = make_trainer(MODEL_NAME, training_args, tokenized_dataset, cartographer=cartographer, num_labels=len(label_set))
        trainer.train()
        #trainer.model = make_model(MODEL_NAME)


    # Saves used Dataset with learnability, variability and confidence of each sample
    cartographer.save_map(f"{args.output_address}_{MODEL_NAME}_e{args.epochs}_r{args.repetitions}_{uuid.uuid4()}")


    print(cartographer.learnability)
