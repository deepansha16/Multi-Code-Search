import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from search_data import *

def tsne_plotter(vectors, query, filename):
    tsne = TSNE(n_components=2, verbose=0, perplexity=2, n_iter=3000)
    tsne_results = tsne.fit_transform(vectors)

    hues = []
    queries = [d["name"] for d in query]
    for query in queries:
        hues.extend([query] * 6)

    df = pd.DataFrame()
    df["x"] = tsne_results[:, 0]
    df["y"] = tsne_results[:, 1]
    plt.figure(figsize=(30, 15))

    sns.scatterplot(
        x="x",
        y="y",
        hue=hues,
        palette=sns.color_palette("bright"),
        data=df,
        legend="full",
        alpha=1.0,
    )
    plt.savefig(f"data/figures/{filename}.png")



def get_ground_truths(path):
    classes = []
    with open(path) as f:
        data = f.read().split("\n\n")
        for d in data:
            d = d.split("\n")
            classes.append({"query": d[0], "name": d[1], "file": d[2]})
    return classes


def get_precision(truth, results):
    row_found = 1
    for _, row in results.iterrows():
        if row["name"] == truth["name"]:
            return 1 / row_found
        row_found += 1
    return 0


def get_recall(precision):
    return int(precision > 0)

def precision_recall(ground_truth, df):
    scores = {"FREQ": [], "TFIDF": [], "LSI": [], "DOC2VEC": []}
    vectors = {"LSI": [], "DOC2VEC": []}
    i = 0
    for truth in ground_truth:
        print(i)
        i += 1
        corpus, dictionary, bow, df = train_data()
        results, vector = get_results(truth["query"], corpus, dictionary, bow, df)
        vectors["LSI"] += [[truth["query"], vector["LSI"]]]
        vectors["DOC2VEC"] += [[truth["query"], vector["DOC2VEC"]]]
        for type in ["FREQ", "TFIDF", "LSI", "DOC2VEC"]:
            precision = get_precision(truth, results[type])
            recall = get_recall(precision)
            scores[type].append((precision, recall))
    return scores, vectors


def get_mean(prec_recs):
    precisions = [i[0] for i in prec_recs]
    recalls = [i[1] for i in prec_recs]
    l = len(prec_recs)
    return sum(precisions) / l, sum(recalls) / l


def print_scores(results):
    for key in results:
        print(key.center(80, "-"))
        prec, rec = get_mean(results[key])
        print(f"\tPrecision: {prec}\n\tRecall: {rec}")
        print("-" * 80, "\n\n")


def plot_queries(vectors, truths):
    vec1 = vectors["LSI"]
    vec2 = vectors["DOC2VEC"]
    vec12 = []
    vec22 = []
    for i in vec1:
        vec12.extend(i[1])
    for i in vec2:
        vec22.extend(i[1])
    tsne_plotter(vec12, truths, "lsi-plot")
    tsne_plotter(vec22, truths, "doc2vec-plot")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python prec_recall.py /path/to/ground_truth_file.txt")
        sys.exit(1)
    query = sys.argv[1]
    df = pd.read_csv("./data/data.csv")
    truths = get_ground_truths(query)
    gg = precision_recall(truths, df)
    print_scores(gg[0])
    print(plot_queries(gg[1], truths))
