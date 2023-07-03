import torch
from transformers import MBartForConditionalGeneration, MBart50Tokenizer

import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
translate_model = MBartForConditionalGeneration.from_pretrained(config.TRANSLATE_MODEL_NAME).to(device)
translate_tokenizer = MBart50Tokenizer.from_pretrained(config.TRANSLATE_MODEL_NAME)


def translate_french_to_english(text):
    return translate(text, source_lang=f'{config.TRANSLATE_LANGUAGE_FRENCH}_XX', target_lang=f'{config.TRANSLATE_LANGUAGE_ENGLISH}_XX')



def translate_english_to_french(text):
    return translate(text, source_lang=f'{config.TRANSLATE_LANGUAGE_ENGLISH}_XX', target_lang=f'{config.TRANSLATE_LANGUAGE_FRENCH}_XX')



def translate(text, source_lang=f'{config.TRANSLATE_LANGUAGE_FRENCH}_XX', target_lang=f'{config.TRANSLATE_LANGUAGE_ENGLISH}_XX'):
    text = text[:min(len(text), config.TRANSLATE_MAX_LENGTH)]
    translate_tokenizer.src_lang = source_lang
    tokenized_text = translate_tokenizer(text, return_tensors='pt').to(device)
    translated_tokens = translate_model.generate(**tokenized_text, forced_bos_token_id=translate_tokenizer.lang_code_to_id[target_lang])
    translation = translate_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    
    if len(translation) > 0:
        return translation[0]
    
    return translation
