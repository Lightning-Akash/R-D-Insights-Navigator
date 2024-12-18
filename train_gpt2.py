import os
import warnings
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    LineByLineTextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore")

def load_dataset(file_path, tokenizer, block_size=128):
    """
    Loads the dataset using LineByLineTextDataset.
    Each line in the file is treated as a separate training example.
    """
    if not os.path.exists(file_path):
        raise ValueError(f"Input file path {file_path} not found")

    print("Loading dataset...")
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
    )
    print(f"Number of samples in dataset: {len(dataset)}")
    return dataset

def train_gpt2(file_path):
    # Check if the input file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The dataset file '{file_path}' does not exist.")

    # Load GPT-2 tokenizer and model
    print("Initializing tokenizer and model...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Padding token fix for GPT-2

    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Load and tokenize the dataset
    dataset = load_dataset(file_path, tokenizer)

    # Create a data collator for padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=3,             # You can increase if needed
        per_device_train_batch_size=4,  # Adjust batch size if needed
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        prediction_loss_only=True,
        evaluation_strategy="no",
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # Train the model
    print("Training started...")
    trainer.train()
    print("Training completed!")

    # Save the fine-tuned model
    print("Saving the model...")
    model.save_pretrained("./fine_tuned_gpt2")
    tokenizer.save_pretrained("./fine_tuned_gpt2")
    print("Model and tokenizer saved to './fine_tuned_gpt2'.")

if __name__ == "__main__":
    dataset_path = r"D:\Aim Here\All Projects Begins\R&D_Project\custom_data.txt"
    train_gpt2(dataset_path)
