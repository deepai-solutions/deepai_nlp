from gensim.models import Word2Vec
import os
__author__ = "Cao Bot"
__copyright__ = "Copyright 2018, DeepAI-Solutions"


class BaseWord2Vec(Word2Vec):
    @classmethod
    def load_model(cls, model_path=None):
        if model_path is None:
            current_file_path = os.path.realpath(__file__)
            current_dir = os.path.dirname(current_file_path)
            parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
            model_path = os.path.join(parent_dir, "models/pretrained_word2vec.bin")
        return cls.load(model_path)


def load_data_from_file(data_path):
    """
    Load data from file
    :param data_path: input file path
    :return: sentences and labels
    Examples
    sentences = [['Hello', 'World'], ['Hello', 'World']]
    """
    sentences = []
    with open(data_path, 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            sent = line.strip().split()
            sentences.append(sent)

    return sentences


def load_data_from_dir(data_path):
    """
    Load data from directory which contains multiple files
    :param data_path: path to data directory
    :return: 2D-array of sentences
    """
    file_names = os.listdir(data_path)
    sentences = None
    for f_name in file_names:
        file_path = os.path.join(data_path, f_name)
        if f_name.startswith('.') or os.path.isdir(file_path):
            continue
        batch_sentences = load_data_from_file(file_path)
        if sentences is None:
            sentences = batch_sentences
        else:
            sentences += batch_sentences
    return sentences


def train(data_path="../data/word_embedding/samples/training", load_data=load_data_from_dir,
          model_path="../models/word2vec.model"):
    """
    Train data loaded from a file or a directory
    :param data_path: path to a file or to a directory which contains multiple files
    :param load_data: function to load data (from a file or a directory)
    :param model_path: path to save model as a file
    :return: None
    """
    sentences = load_data(data_path)
    model = Word2Vec(sentences, size=100, window=5, min_count=1, workers=4)
    model.save(model_path)


def train_multiple_dir(data_path_list=["../data/word_embedding/samples/training"], load_data=load_data_from_dir,
                       model_path="../models/word2vec.model"):
    """
    Train data loaded from a file or a directory
    :param data_path_list: list of paths to a file or to a directory which contains multiple files
    :param load_data: function to load data (from a file or a directory)
    :param model_path: path to save model as a file
    :return: None
    """
    sentences = None
    for data_path in data_path_list:
        if sentences is None:
            sentences = load_data(data_path)
        else:
            sentences += load_data(data_path)
    model = Word2Vec(sentences, size=100, window=5, min_count=1, workers=4)
    model.save(model_path)


def test(model_path="../models/word2vec.model", word="thu_nhập"):
    """
    Test word2vec model
    :param model_path: path to model file
    :param word: word to test
    :return: None
    """
    model = Word2Vec.load(model_path)
    vector = model.wv[word]
    print(vector)
    sim_words = model.wv.most_similar(word)
    print(sim_words)


def word2vec_statistic(input_path, model_path=None):
    """
    Make statistics for missing words
    :param input_path: input file path or directory path
    :param model_path: path to model file
    :return: (miss count, total count)
    """
    if model_path is None:
        current_file_path = os.path.realpath(__file__)
        current_dir = os.path.dirname(current_file_path)
        parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
        model_path = os.path.join(parent_dir, "models/pretrained_word2vec.bin")
    model = Word2Vec.load(model_path)

    def check_words(words):
        count = 0
        miss_count = 0
        for word in words:
            word = word.strip()
            if len(word) == 0:
                continue
            count += 1
            if word.lower() not in model.wv:
                miss_count += 1
        return miss_count, count

    def check_file(file_path):
        with open(file_path, 'r') as fr:
            lines = fr.readlines()
        words = []
        for line in lines:
            words += line.strip().split()
        return check_words(words)
    if os.path.isdir(input_path):
        total_count = 0
        total_miss_count = 0
        file_name_list = os.listdir(input_path)
        file_name_list = [fn for fn in file_name_list if not fn.startswith(".")]
        file_paths = [os.path.join(input_path, fn) for fn in file_name_list]
        file_paths = [fp for fp in file_paths if os.path.isfile(fp)]
        for fp in file_paths:
            miss_, count_ = check_file(fp)
            total_count += count_
            total_miss_count += miss_
        return total_miss_count, total_count
    else:
        return check_file(input_path)


if __name__ == '__main__':
    # train()
    test(model_path="../models/pretrained_word2vec.bin")



