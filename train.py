import json
import os
import random
from typing import Optional
from datasets import load_dataset, load_from_disk
from transformers import GPT2Tokenizer, Trainer, TrainingArguments, GPT2LMHeadModel, GPT2Config
from pprint import pprint
from model import UrduALGPT2LMHeadModel

DEFAULT_MODEL_NAME = "gpt2"

save_path = "./"

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate_post_training(trainer: Trainer, dataset: dict, save_path: str) -> dict:
    # Evaluate the model
    trainer_evaluation_result = trainer.evaluate()
    # Compute perplexity
    perplexity = evaluate.load("perplexity", module_type="metric")
    input_texts = [s for s in dataset['test']['text'] if s != '']
    results = perplexity.compute(model_id=save_path, predictions=input_texts)
    trainer_evaluation_result['test_mean_perplexity'] = results['mean_perplexity']
    pprint(trainer_evaluation_result)
    return trainer_evaluation_result

def split_dataset(dataset, train_percentage, val_percentage):
    total_length = len(dataset)
    train_length = int(train_percentage * total_length)
    val_length = int(val_percentage * total_length)
    
    train_dataset = dataset.select(range(train_length))
    val_dataset = dataset.select(range(train_length, train_length + val_length))
    test_dataset = dataset.select(range(train_length + val_length, total_length))
    
    return train_dataset, val_dataset, test_dataset

def run(model_class_name: str, model_name: str = DEFAULT_MODEL_NAME, minimize_dataset: bool = False,
        pretrained: bool = False, depth: Optional[int] = None, batch_size: int = 32,
        num_of_epochs: float = 1.0, load_checkpoint: bool = False, dataset_path: str = "wikitext-103-raw-v1",
        sequence_max_length: int = 512, learning_rate: float = 1e-5, device="gpu"):
    # Load a small dataset from hugging face
    assert device.lower() in ["gpu", "tpu", "cpu"]

    dataset = load_dataset("anuragshas/ur_opus100_processed")

    if minimize_dataset:
        dataset = dataset['train']

    # Split dataset into train, validation, and test sets
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, train_percentage=0.8, val_percentage=0.1)

    print("Train dataset size:", len(train_dataset))
    print("Validation dataset size:", len(val_dataset))
    print("Test dataset size:", len(test_dataset))

    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Set the padding token for the tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_class = {'GPT2LMHeadModel': GPT2LMHeadModel, 'ALGPT2LMHeadModel': ALGPT2LMHeadModel}[model_class_name]
    if pretrained:
        model = model_class.from_pretrained(model_name)
    else:
        config = GPT2Config(vocab_size=tokenizer.vocab_size) if depth is None else GPT2Config(
            vocab_size=tokenizer.vocab_size, n_layer=depth)
        model = model_class(config)
    print(model)
    print("Number of parameters:", count_parameters(model))

    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=sequence_max_length)

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

    # Add labels for the language modeling task
    tokenized_train_dataset = tokenized_train_dataset.map(lambda examples: {'labels': examples['input_ids']}, batched=True)
    tokenized_val_dataset = tokenized_val_dataset.map(lambda examples: {'labels': examples['input_ids']}, batched=True)
    tokenized_test_dataset = tokenized_test_dataset.map(lambda examples: {'labels': examples['input_ids']}, batched=True)

    # Define training arguments and initialize Trainer
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=10,
        per_device_eval_batch_size=16,
        num_train_epochs=num_of_epochs,
        logging_dir='./logs',
        logging_steps=100,
        save_strategy='steps',
        save_steps=10000 if not minimize_dataset else 10,
        learning_rate=learning_rate,
        evaluation_strategy='steps',
        eval_steps=10000 if not minimize_dataset else 10,
    )

    trainer = Trainer(
        model=model if device.lower() != "tpu" else model.to(device),
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        tokenizer=tokenizer
    )

    full_path = f"save_{model_class_name}-{depth}"
    # Start training
    trainer.train(resume_from_checkpoint=full_path) if load_checkpoint else trainer.train()

    # Save the model
    trainer.save_model(full_path)
    # Evaluate the model on the test dataset
    trainer_evaluation_result = trainer.evaluate(eval_dataset=tokenized_test_dataset)
    pprint(trainer_evaluation_result)
    with open(f"{full_path}/eval_results.json", 'w') as f:
        json.dump(trainer_evaluation_result, f)

if __name__ == '__main__':
    run(model_class_name='GPT2LMHeadModel', minimize_dataset=True, pretrained=False, depth=3, load_checkpoint=False,
        num_of_epochs=0.5)
