import pandas as pd

from translate import translate_french_to_english, translate_english_to_french
from summarize import summarize
from similarity import similarity


def evaluate_row(text, heading):
    translated_text = translate_french_to_english(text)
    translated_text_heading = summarize(translated_text)    
    source_lang_translated_text_heading = translate_english_to_french(translated_text_heading) 
    similarity_score = similarity(heading, source_lang_translated_text_heading)
    return similarity_score



def evaluate(df):
    scores = []

    for index, row in df.iterrows():
        score = evaluate_row(row['maintext'], row['title'])
        scores.append(score)
    
    df['scores'] = scores

    df.to_csv('./results/score.csv', index=False)
    


def get_data():
    df = pd.read_csv('./data/fr_covid_news_8k.csv', usecols=['maintext', 'title'])
    df = df.sample(n=500, replace=True)
    df = df.reset_index(drop=True)
    return df



if __name__ == '__main__':
    df = get_data()
    evaluate(df)