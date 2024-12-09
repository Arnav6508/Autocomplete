def count_n_grams(data, n, start_token = '<s>', end_token = '<e>'):
    n_grams = {}
    for sentence in data:
        sentence = (start_token,)*n + tuple(sentence) +(end_token,)

        for i in range(len(sentence)-n+1):
            n_gram = sentence[i:i+n]
            n_grams[n_gram] = n_grams.get(n_gram,0) + 1

    return n_grams

def estimate_prob_for_one(word, prev_n_gram, n_gram_counts, n_plus1_gram_counts, vocab_size, k = 1):
    prev_n_gram = tuple(prev_n_gram)
    n_plus1_gram = prev_n_gram + (word,)

    numerator = n_plus1_gram_counts.get(n_plus1_gram,0) + k
    denominator = n_gram_counts.get(prev_n_gram,0) + k*vocab_size
    return numerator/denominator

def estimate_prob(prev_n_gram, n_gram_counts, n_plus1_gram_counts, vocab, end_token='<e>', unknown_token="<unk>", k = 1):
    probs = {}
    vocab[unknown_token] = 1
    vocab[end_token] = 1
    vocab_size = len(vocab)

    for word in vocab.keys():
        prob = estimate_prob_for_one(word, prev_n_gram, n_gram_counts, n_plus1_gram_counts, vocab_size, k)
        probs[word] = prob
    
    return probs

def perplexity_one_sentence(sentence, n_gram_counts, n_plus1_gram_counts, vocab, start_token='<s>', end_token = '<e>', k=1):
    N = len(sentence)
    n = len(list(n_gram_counts.keys())[0])

    sentence = [start_token] * n + sentence + [end_token]
    prob = 1

    for i in range(N-n+1):
        prev_n_gram = sentence[i:i+n]
        word = sentence[i+n]
        prob *= 1/estimate_prob_for_one(word, prev_n_gram, n_gram_counts, n_plus1_gram_counts, len(vocab), k)
    
    perplexity = prob**(1/N)
    return perplexity

def perplexity(data, n_gram_counts, n_plus1_gram_counts, vocab, uknown_token='<unk>', end_token = '<e>', k=1):
    perplexity = 0
    for sentence in data: 
        perplexity += perplexity_one_sentence(sentence, n_gram_counts, n_plus1_gram_counts, vocab, uknown_token, end_token, k)
    return perplexity/len(data)
