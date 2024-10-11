# src/fine_tuning.py

import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from src.config import  MODEL_PATH, FINE_TUNE_PARAMS,MAX_SQ_LENGTH
import src.model_finetuning.data_loader as data_loader,src.model_finetuning.model_loader as model_loader
def fine_tune_model():
    model,tokenizer = model_loader()
    dataset = data_loader(tokenizer)  # Load your fine-tuning dataset here

    # Set training arguments
    training_args = TrainingArguments(
        output_dir=MODEL_PATH,
        per_device_train_batch_size=FINE_TUNE_PARAMS['batch_size'],
        num_train_epochs=FINE_TUNE_PARAMS['num_train_epochs'],
        learning_rate=FINE_TUNE_PARAMS['learning_rate'],
        weight_decay=FINE_TUNE_PARAMS['weight_decay'],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs',
    )

    trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length =MAX_SQ_LENGTH ,
    dataset_num_proc = 2,
    args = training_args,
)

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model(MODEL_PATH)
