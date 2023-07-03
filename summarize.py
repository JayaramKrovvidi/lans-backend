import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import nltk

import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
summary_model = AutoModelForSeq2SeqLM.from_pretrained(config.SUMMARY_MODEL_PATH).to(device)
summary_tokenizer = AutoTokenizer.from_pretrained(config.SUMMARY_MODEL_PATH)



def summarize(text):
    input = [config.SUMMARY_PREFIX + text]

    tokenized_input = summary_tokenizer(input, max_length=config.SUMMARY_MAX_INPUT_LENGTH, truncation=True, return_tensors='pt').to(device)
    output = summary_model.generate(**tokenized_input, num_beams=8, do_sample=True, min_length=config.SUMMARY_MIN_OUTPUT_LENGTH, max_length=config.SUMMARY_MAX_OUTPUT_LENGTH)
    tokenized_output = summary_tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    summarize_title = nltk.sent_tokenize(tokenized_output.strip())[0]

    return summarize_title

   