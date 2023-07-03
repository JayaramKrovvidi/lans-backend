TRANSLATE_MODEL_NAME='facebook/mbart-large-50-many-to-many-mmt'
TRANSLATE_LANGUAGE_ENGLISH='en'
TRANSLATE_LANGUAGE_FRENCH='fr'
TRANSLATE_MAX_LENGTH=3000

SUMMARY_MODEL_NAME='t5-small'
SUMMARY_MODEL_PATH='models/t5-small-summarize'
SUMMARY_PREFIX  = 'summarize: '
SUMMARY_MAX_INPUT_LENGTH=512
SUMMARY_MIN_OUTPUT_LENGTH=200
SUMMARY_MAX_OUTPUT_LENGTH=500

SIMILARITY_MODEL_NAME='stsb-roberta-large'

RANDOM_SEED=1234