import random
from preprocess import preprocess, tokenize

def load_data(data_file, training_split, count_threshold = 2, unknown_token = "<unk>"):
    with open(data_file, 'r') as file:
        data = file.read()
    
    data = tokenize(data)
    data_size = len(data)
    train_size = int(data_size*training_split)
    
    random.seed(87)
    random.shuffle(data)

    training_data = data[:train_size]
    testing_data = data[train_size:]

    training_data, testing_data, vocab = preprocess(training_data, testing_data, count_threshold, unknown_token)

    return training_data, testing_data, vocab