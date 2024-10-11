from unsloth import FastLanguageModel
import config

def load_dataset():
    model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = config.MODEL_NAME, # Reminder we support ANY Hugging Face model!
    max_seq_length = config.MAX_SQ_LENGTH,
    dtype = config.DTYPE,
    load_in_4bit = config.LOAD_IN_4_BITS,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    model = FastLanguageModel.get_peft_model(
    model,
    r=config.PEFT_PARAMS['lora_r'],                  # Low-rank approximation for LoRA
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=config.PEFT_PARAMS['lora_alpha'],     # Scaling factor for LoRA
    lora_dropout=config.PEFT_PARAMS['lora_dropout'], # Dropout rate for LoRA layers
    bias="none",                               # No bias for LoRA
    use_gradient_checkpointing="unsloth",     # Optimized for long contexts
    random_state=3407,
    use_rslora=False,                          # Disable rank stabilized LoRA
    loftq_config=None                          # LoftQ configuration
)
)
    return model,tokenizer