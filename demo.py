from deepai_nlp.tokenization.crf_tokenizer import CrfTokenizer
from deepai_nlp.word_embedding.word2vec_gensim import BaseWord2Vec
from deepai_nlp.text_classification.short_text_classifiers import BiDirectionalLSTMClassifier, load_synonym_dict

tokenizer = CrfTokenizer()
word2vec_model = BaseWord2Vec.load_model()

sym_dict = load_synonym_dict('deepai_nlp/data/sentiment/synonym.txt')
keras_text_classifier = BiDirectionalLSTMClassifier(tokenizer=tokenizer, word2vec=word2vec_model.wv,
                                                    model_path='deepai_nlp/models/sentiment_model.h5',
                                                    max_length=10, n_epochs=10,
                                                    sym_dict=sym_dict)
# Load and prepare data
X, y = keras_text_classifier.load_data(['deepai_nlp/data/sentiment/samples/positive.txt',
                                       'deepai_nlp/data/sentiment/samples/negative.txt'],
                                       load_method=keras_text_classifier.load_data_from_file)

# Train your classifier and test the model
keras_text_classifier.train(X, y)
label_dict = {0: 'tích cực', 1: 'tiêu cực'}
test_sentences = ['Dở thế', 'Hay thế', 'phim chán thật', 'nhảm quá']
labels = keras_text_classifier.classify(test_sentences, label_dict=label_dict)
print(labels)  # Output: ['tiêu cực', 'tích cực', 'tiêu cực', 'tiêu cực']
