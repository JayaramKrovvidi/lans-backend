import argparse
import json
import numpy as np

from sentence_transformers import SentenceTransformer

import config

model = SentenceTransformer(config.SUMMARY_MODEL_PATH)

np.random.seed(config.RANDOM_SEED)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def unit_vector(vec):
    return vec / np.linalg.norm(vec)



def cos_sim(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.clip(np.tensordot(v1_u, v2_u, axes=(-1, -1)), -1.0, 1.0)



def weat_association(W, A, B):
    return np.mean(cos_sim(W, A), axis=-1) - np.mean(cos_sim(W, B), axis=-1)



def weat_score(X, Y, A, B):
    x_association = weat_association(X, A, B)
    y_association = weat_association(Y, A, B)

    tmp1 = np.mean(x_association, axis=-1) - np.mean(y_association, axis=-1)
    tmp2 = np.std(np.concatenate((x_association, y_association), axis=0))

    return tmp1 / tmp2



def balance_word_vectors(vec1, vec2):
    diff = len(vec1) - len(vec2)

    if diff > 0:
        vec1 = np.delete(vec1, np.random.choice(len(vec1), diff, 0), axis=0)
    else:
        vec2 = np.delete(vec2, np.random.choice(len(vec2), -diff, 0), axis=0)

    return (vec1, vec2)



def get_word_vectors(words):
    outputs = []
    
    for word in words:
        try:
            outputs.append(model.encode([word])[0])
        except:
            pass

    return np.array(outputs)



def compute_weat():
    weat_path = './data/weat.json'

    with open(weat_path) as f:
        weat_dict = json.load(f)

    all_scores = {}

    for data_name, data_dict in weat_dict.items():
        # Target
        X_key = data_dict['X_key']
        Y_key = data_dict['Y_key']
        
        # Attributes
        A_key = data_dict['A_key']
        B_key = data_dict['B_key']

        X = get_word_vectors(data_dict[X_key])
        Y = get_word_vectors(data_dict[Y_key])
        A = get_word_vectors(data_dict[A_key])
        B = get_word_vectors(data_dict[B_key])

        if len(X) == 0 or len(Y) == 0:
            print('Not enough matching words in dictionary')
            continue

        X, Y = balance_word_vectors(X, Y)
        A, B = balance_word_vectors(A, B)

        score = weat_score(X, Y, A, B)
        all_scores[data_name] = str(score)

    return all_scores



def dump_dict(obj):
    with open('./results/bias-results.json', "w") as file:
        json.dump(obj, file)



if __name__ == '__main__':
    bias_score = compute_weat()

    print("Final Bias Scores")
    print(json.dumps(bias_score, indent=4))

    dump_dict(bias_score)