from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from itertools import chain,repeat,islice
from torchnlp.word_to_vector import GloVe

class TextProcess:
    """
    process corpus
        1、tokenize
        2、stop word
        3、lemmatizer
        4、fiexd length
        5、token to index
        5、token embedding matrix
    """
    def __init__(self):
        self.stop_words = stopwords.words("english")
        self.glove_vectors = GloVe(name='6B')
        self.lemma = WordNetLemmatizer()
        self.token_index = {}
        return

    def tokenize(self, sentence):
        """
        tokenizer and stop word and lemmatizer
        """
        return [self.lemma.lemmatize(token.lower()) for token in word_tokenize(sentence) if token not in self.stop_words]

    def padding(self, sentence, max_length, padding_value=0):
        """
        fiexd length
        """
        return list(islice(chain(sentence, repeat(padding_value)), max_length))

    def encoder_corpus(self, corpus):

        corpus_token = list(map(self.tokenize, corpus))

        # create token to index dict
        for index, token in enumerate(set(chain(*corpus_token)), start=1):
            self.token_index[token] = index

        # lambda token to index
        lambda_token_to_index = lambda tokens: list(map(self.token_index.get, tokens))
        corpus_index = list(map(token_to_index, corpus_token))

        # get max sentence length
        max_length = max([len(token_indexes) for token_indexes in corpus_index])

        # lambda padding sentence with 0
        lambda_sentence_padding = lambda tokens: self.padding(tokens, max_length)
        return list(map(sentence_padding, corpus_index))

    def get_embedding(self):
        self.token_index = dict(sorted(self.token_index.items(), key=lambda x: x[1]))

        # lambda get token embedding
        lambda_token_embedding  = lambda token: self.glove_vectors[token]
        token_embedding = map(lambda_token_embedding, list(self.token_index.keys()))

        # embedding matrix
        pre_trained_embedding = np.array([embedding_tensor.numpy() for embedding_tensor in token_embedding])
        return pre_trained_embedding
