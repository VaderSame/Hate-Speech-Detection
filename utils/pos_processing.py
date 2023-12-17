# utils/pos_processing.py
<<<<<<< Updated upstream
=======

>>>>>>> Stashed changes
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.text_processing import preprocess, basic_tokenize

def get_pos_tags(tweets):
    tweet_tags = []
    for t in tweets:
        tokens = basic_tokenize(preprocess(t))
        tags = nltk.pos_tag(tokens)
        tag_list = [x[1] for x in tags]
        tag_str = " ".join(tag_list)
        tweet_tags.append(tag_str)
    return tweet_tags

def get_pos_vectorizer():
    return TfidfVectorizer(
        tokenizer=None,
        lowercase=False,
        preprocessor=None,
        ngram_range=(1, 3),
        stop_words=None,
        use_idf=False,
        smooth_idf=False,
        norm=None,
        decode_error='replace',
        max_features=5000,
        min_df=5,
        max_df=0.75,
    )
<<<<<<< Updated upstream

# def get_pos_matrix(tweet_tags):
#     pos_vectorizer = get_pos_vectorizer()
#     pos = pos_vectorizer.fit_transform(pd.Series(tweet_tags)).toarray()
#     pos_vocab = {v: i for i, v in enumerate(pos_vectorizer.get_feature_names_out())}
#     return pos, pos_vocab
=======
>>>>>>> Stashed changes
