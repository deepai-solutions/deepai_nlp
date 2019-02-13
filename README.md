pip install -e .

### Load pre-trained tokenizer
```python
from deepai_nlp.tokenization.crf_tokenizer import CrfTokenizer
tokenizer = CrfTokenizer()
```

### Load pre-trained word2vec
```python
from deepai_nlp.word_embedding.word2vec_gensim import BaseWord2Vec
word2vec_model = BaseWord2Vec.load_model()
```

### Load text classifier
```python
from deepai_nlp.text_classification.short_text_classifiers import BiDirectionalLSTMClassifier, load_synonym_dict
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
```

### Crawl wiki article:
```python
from wikicrawler.wiki_bs4 import WikiTextCrawler
wiki_crawler = WikiTextCrawler()
keywords = ['Học máy'] 
for word in keywords:
results = wiki_crawler.search(word)
sample_url = results[0]
wiki_crawler.write_text(output_file='wiki.txt', url=sample_url, mode='w')
```

## For training tokenizer and word2vec

Please see file: deepai_nlp/README.md
