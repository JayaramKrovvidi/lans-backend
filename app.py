from flask import Flask, request, jsonify

from translate import translate_french_to_english, translate_english_to_french
from summarize import summarize
from similarity import similarity

import config


app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    source_lang = request.form["source_lang"]
    text = request.form["text"]
    heading = request.form["heading"]
    
    translated_text = text
    source_lang_translated_text_heading = heading
    
    if source_lang == config.TRANSLATE_LANGUAGE_FRENCH:
        translated_text = translate_french_to_english(text)

    translated_text_heading = summarize(translated_text)    

    if source_lang == config.TRANSLATE_LANGUAGE_FRENCH:
        source_lang_translated_text_heading = translate_english_to_french(translated_text_heading)
    
    similarity_score = similarity(heading, source_lang_translated_text_heading)
    
    return jsonify({
                        'translated_heading': translated_text_heading, 
                        'source_translated_heading': source_lang_translated_text_heading, 
                        'similarity_score': similarity_score
                    })

app.run()
