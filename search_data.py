import os
import pathlib
import pickle
import re
import string
import sys
from collections import defaultdict
from typing import List

import pandas as pd
import gensim
from gensim.corpora import Dictionary
from gensim.models import Doc2Vec, LsiModel, TfidfModel, doc2vec
from gensim.models.doc2vec import TaggedDocument
from gensim.similarities import MatrixSimilarity, SparseMatrixSimilarity
from gensim.utils import simple_preprocess

# Define Stopwords
STOPWORDS = ["this", "def", "self", "todo", "test", "main", "tests"]

# Splitting of names by camel-case and underscore
def split_camelcase(name: str) -> List[str]:
    return re.sub(r"((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))", r" \1", name).split(" ")  # Define regex

def split_camelcases(tokens: List[str]) -> List[str]:
    return [split_camelcase(tok) for tok in tokens]

def split_underscore(tokens: List[str]) -> List[str]:
    w = [word for token in tokens for word in token.split("_")]
    return w

def split_space(text: str) -> str:
    text = text if str(text) != "nan" else ""
    filter = "[\[\]\.\\/:\,\\\"\$\#\%\`\]\(\)\*\-]"
    text = re.sub(filter, " ", text)
    text = re.sub("\s{2,}", " ", text)
    prep = simple_preprocess(text)
    return prep

def remove_stop(tokens: List[str]) -> List[str]:
    cleaned = []
    for tok in tokens:
        if tok not in STOPWORDS:
            cleaned.append(tok)
    return cleaned

def load_data(file: pathlib.Path) -> pd.DataFrame:
    return pd.read_csv(file)

def lowercase(tokens) -> List[str]:
    out = []
    for token in tokens:
        b = [i.lower() for i in token]
        out.extend(b)
    return out

def get_bow(tokens, frequency):
    corpus = [[token for token in text if frequency[token] > 2] for text in tokens]
    dictionary = Dictionary(corpus)
    bow = [dictionary.doc2bow(text) for text in corpus]
    return corpus, dictionary, bow


def serialize_data(data, name) -> None:
    pickle.dump(
        data, open(f"data/bin/{name}.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL
    )
def load_serialized(name):
    return pickle.load(open(f"data/bin/{name}.pkl", "rb"))

# Main preprocessing function
def preprocess(df: pd.DataFrame):
    tokens = [
        remove_stop(
            lowercase(
                split_camelcases(
                    split_underscore([row["name"]] + split_space(row["comment"]))
                )
            )
        )
        for _, row in df.iterrows()
    ]
    frequency = defaultdict(int)
    for token in tokens:
        for word in token:
            frequency[word] += 1
    corpus, dictionary, bow = get_bow(tokens, frequency)

    serialize_data(corpus, "corpus")
    serialize_data(dictionary, "dictionary")
    serialize_data(bow, "bow")

    return (corpus, dictionary, bow)

def make_frequency(bow, dictionary: Dictionary) -> SparseMatrixSimilarity:
    model = SparseMatrixSimilarity(bow, len(dictionary.token2id))
    serialize_data(model, "frequency_model")
    return model


def make_tfidf(bow) -> TfidfModel:
    model = TfidfModel(bow)
    serialize_data(model, "tfidf_model")
    return model


def make_lsi(bow, corpus, dictionary: Dictionary) -> LsiModel:
    model = LsiModel(bow, id2word=dictionary, num_topics=300)
    serialize_data(model, "lsi_model")

    return model


def tagged_document(list_of_list_of_words):
    for i, list_of_words in enumerate(list_of_list_of_words):
        yield doc2vec.TaggedDocument(list_of_words, [i])


def make_doc2vec(corpus) -> Doc2Vec:
    training_data = list(tagged_document(corpus))
    model = Doc2Vec(vector_size=150, epochs=25, workers=3)
    model.build_vocab(training_data)
    model.train(training_data, total_examples=model.corpus_count, epochs=model.epochs)
    serialize_data(model, "doc2vec_model")
    return model


def load_model(name: str):
    path = f"data/bin/{name}_model.pkl"
    if not os.path.exists(path):
        return
    with open(path, "rb") as f:
        return pickle.load(f)


def query_tfidf_model(query: str, bow, dictionary: Dictionary, n=5):
    tfidf = load_model("tfidf")
    if not tfidf:
        tfidf = make_tfidf(bow)
    index = SparseMatrixSimilarity(bow, len(dictionary.token2id))
    return predict_topn(query, dictionary, n, tfidf, index)


def query_freq_model(query: str, bow, dictionary: Dictionary, n=5):
    index = load_model("frequency")
    query_document = query.split()
    query_bow = dictionary.doc2bow(query_document)
    simils = index[query_bow]
    top_n = sorted(list(enumerate(simils)), key=lambda x: x[1], reverse=True)[:n]
    return [i[0] for i in top_n], ""


def query_lsi_model(query: str, bow, dictionary: Dictionary, n=5):
    lsi = load_model("lsi")
    index = MatrixSimilarity(lsi[bow])
    top, query_vect = predict_topn(query, dictionary, n, lsi, index)
    embedding = [[value for _, value in query_vect]] + [
        [value for _, value in lsi[bow][i]] for i in top
    ]
    return top, embedding

def query_doc2vec_model(query: str, corpus, n=5):
    doc2vec = load_model("doc2vec")
    embed = doc2vec.infer_vector(query.split(" "))
    indices = [i[0] for i in doc2vec.docvecs.most_similar([embed], topn=n)]
    embeddings = [list(embed)] + [
        list(doc2vec.infer_vector(corpus[index])) for index in indices
    ]
    return indices, embeddings
def predict_topn(query, dictionary, n, tfidf, index):
    query_document = query.split()
    query_bow = dictionary.doc2bow(query_document)
    simils = index[tfidf[query_bow]]
    top_n = sorted(list(enumerate(simils)), key=lambda x: x[1], reverse=True)[:n]

    return [i[0] for i in top_n], tfidf[query_bow]

def train_data(file="data/data.csv"):
    df = load_data(file)
    corpus, dictionary, bow = preprocess(df)
    make_frequency(bow, dictionary)
    make_tfidf(bow)
    make_lsi(bow, corpus, dictionary)
    make_doc2vec(corpus)

    return corpus, dictionary, bow, df

def query_model(query_function, args, df):
    top_5_indices, vector = query_function(*args)
    return df.iloc[top_5_indices], vector


def get_results(query, corpus, dictionary, bow, df):
    results = {}
    vectors = {}
    results["FREQ"] = query_model(query_freq_model, (query, bow, dictionary), df)[0]
    results["TFIDF"] = query_model(query_tfidf_model, (query, bow, dictionary), df)[0]
    results["LSI"], vectors["LSI"] = query_model(
        query_lsi_model, (query, bow, dictionary), df
    )
    results["DOC2VEC"], vectors["DOC2VEC"] = query_model(
        query_doc2vec_model, (query, corpus), df
    )
    return results, vectors

def print_top_results(results):
    for type in results:
        print('-*-'*30+'\n')
        print(f"\t{type} Top-5 most similar entities".center(20, ' '))
        print('\n' + '-*-'*30 + '\n')
        data = results[type]
        index = 1
        for _, i in data.iterrows():
            print(f"\n{index}\t Python {i['type']}: {i['name']}")
            print(f" \t File {i['path']}")
            print(f" \t Line {i['line']}")
            print(f" \t Comment: {i['comment']}")
            index += 1
            print('\n' + '---'*30)
        print("\n\n\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: search_data.py search_query")
        sys.exit(1)
    query = sys.argv[1].lower()
    corpus, dictionary, bow, df = train_data()

    print_top_results(get_results(query, corpus, dictionary, bow, df)[0])