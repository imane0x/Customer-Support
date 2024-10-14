# config.py

# Paths
RAW_DATA_PATH = "data/Amazon_Sale_Report.db"   # Path to the raw customer support data
FINE_TUNE_DATA_PATH = "data/fine_tune_data.json" # Path to the prepared fine-tuning dataset
LOGS_PATH = "logs"                               # Path to save logs
DB_PATH = "data/Amazon_Sale_Report.db"
# Data split parameters
TEST_SIZE = 0.3                                   # Proportion of the dataset to include in the test split
RANDOM_STATE = 42                                 # Seed for reproducibility


MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"  # Path to save/load the fine-tuned model
DATASET_NAME = "Kaludi/Customer-Support-Responses"
MAX_SQ_LENGTH = 2048 # Choose any! We auto support RoPE Scaling internally!
DTYPE = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
LOAD_IN_4_BITS = False # Use 4bit quantization to reduce memory usage. Can be False.
MODEL_PATH=  "outputs"

# Fine-tuning parameters
FINE_TUNE_PARAMS = {
    'learning_rate': 5e-5,                       # Learning rate for the optimizer
    'num_train_epochs': 3,                        # Number of training epochs
    'batch_size': 8,                              # Batch size for training
    'weight_decay': 0.01                          # Weight decay for regularization
}

# PEFT (Parameter Efficient Fine-Tuning) parameters
PEFT_PARAMS = {
    'lora_r': 16,                                 # Low-rank approximation for LoRA
    'lora_alpha': 32,                             # Scaling factor for LoRA
    'lora_dropout': 0.1,                          # Dropout rate for LoRA layers
}

# Model Evaluation parameters
EVALUATION_QUERIES = [
    "Where is my order?",
    "Can I cancel my recent order?",
    "What is the status of my order #12345?",
]  # Sample queries for evaluating the fine-tuned model

FINE_TUNED_MODEL = "im21/Customer_Support_Mistral"
