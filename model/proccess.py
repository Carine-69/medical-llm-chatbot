from datasets import load_dataset
from transformers import AutoTokenizer

# Load and format dataset

ds = load_dataset("medalpaca/medical_meadow_medical_flashcards")

def format_row(row):
    return {
        "text": "Instruction: " + row['instruction'] + \
                " Question: " + row['input'] + \
                " Response: " + row['output']
    }

formatted_ds = ds["train"].map(format_row)

# Tokenizer/model loading and LoRA setup should be done in the notebook, not here.
# Provide a function to tokenize with a passed-in tokenizer

def tokenize_row(batch, tokenizer):
    tokens = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

# Export only the formatted dataset
__all__ = ["formatted_ds", "tokenize_row"]


