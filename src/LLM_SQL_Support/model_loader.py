from langchain_huggingface import HuggingFacePipeline
import torch
from config import FINE_TUNED_MODEL

def load_model():
    llm = HuggingFacePipeline.from_model_id(
        model_id =FINE_TUNED_MODEL ,
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 1000},
    )
    return llm