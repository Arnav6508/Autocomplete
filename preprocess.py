import nltk

def split_into_sentences(data):
    sentences = data.split('\n')
    sentences = [s.strip() for s in sentences]
    sentences = [s for s in sentences if len(s)>0]
    return sentences

def tokenize(data):
    tokenize = []
    sentences = split_into_sentences(data)
    for sentence in sentences:
        sentence = sentence.lower()
        tokenized_sentence = nltk.word_tokenize(sentence)
        tokenize.append(tokenized_sentence)
    return tokenize

def create_vocab(tokenized_sentences, count_threshold):
    vocab = {}
    for sentence in tokenized_sentences:
        for word in sentence:
            vocab[word] = vocab.get(word,0)+1

    real_vocab = {}
    for word, count in vocab.items():
        if count>=count_threshold: real_vocab[word] = count
    
    return real_vocab

# oov = out of vocab words (need to be replaced with unknown token)
def replace_oov(tokenized_sentences, vocab, unknown_token):
    for i,sentence in enumerate(tokenized_sentences):
        for j,word in enumerate(sentence):
            if vocab.get(word,0) == 0: tokenized_sentences[i][j] = unknown_token
    return tokenized_sentences

def preprocess(training_data, testing_data, count_threshold, unknown_token):
    vocab = create_vocab(training_data, count_threshold)
    
    training_data = replace_oov(training_data, vocab, unknown_token)
    testing_data = replace_oov(testing_data, vocab, unknown_token)

    return training_data, testing_data, vocab