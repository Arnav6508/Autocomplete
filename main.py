import pickle
from load import load_data
from utils import count_n_grams, estimate_prob, perplexity_one_sentence

def build_model(data_file):
    training_data, testing_data, vocab = load_data(data_file, 1)

    n_grams = {}
    for i in range(1,5):
        n_grams[f'n_{i}_gram'] = count_n_grams(training_data, i)

    with open('./model_weights/n_grams.pkl', 'wb') as file:
        pickle.dump(n_grams, file)

    with open('./model_weights/vocab.pkl', 'wb') as file:
        pickle.dump(vocab, file)

    print('Vocab and n_grams stores successfully')


def suggest_a_word(prev_tokens, n, start_token = '<s>', end_token='<e>', unknown_token="<unk>", k=1.0, starts_with = None):

    if n != 2 and n != 3: 
        print('Enter 2 or 3 for bigram or trigram based model')
        return

    with open('./model_weights/n_grams.pkl','rb') as file:
        n_grams = pickle.load(file)

    n_gram_counts = n_grams[f'n_{n}_gram']
    n_plus1_gram_counts = n_grams[f'n_{n+1}_gram']

    with open('./model_weights/vocab.pkl','rb') as file:
        vocab = pickle.load(file)

    prev_tokens = [start_token]*n + prev_tokens
    
    prev_n_gram = prev_tokens[-n:]
    probs = estimate_prob(prev_n_gram, n_gram_counts, n_plus1_gram_counts, vocab)

    max_prob = 0
    best_word = None
    for word, prob in probs.items():
        if starts_with and not word.startswith(starts_with): continue
        if prob>max_prob: 
            max_prob = prob
            best_word = word

    print(best_word, max_prob)
    return best_word, max_prob


