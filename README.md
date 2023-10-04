# Multi-Code-Search

## Goal

Develop a search engine that can query a large Python code repository using multiple sources of information.

## Steps

1. **Extract Data:** Extract names/comments of Python classes, methods, and functions.

2. **Train Search Engines:** Represent code entities using four embeddings: frequency, TF-IDF, LSI, and Doc2Vec. Report the entities most similar to the given query string.

3. **Evaluate Search Engines:** Define the ground truth for a set of queries and measure average precision and recall for the four search engines.

4. **Visualize Query Results:** For LSI and Doc2Vec, project the embedding vectors of queries and the top-5 answers to a 2D plot using t-SNE.

## Methodology of Training

Create a corpus from the code entity names and comments:

1. Split entity names by camel-case and underscore (e.g., go_to_myHome -> [go, to, my, home]).

2. Filter stopwords = {test, tests, main, this,..} (enlarge list with appropriate candidates).

3. Convert all words to lowercase.

4. Analyze whether you need the whole comment for training or if cleaning (e.g., removing code snippets)/using only a part of it could be beneficial for performance.

Represent entities using the following vector embeddings:

- FREQ: frequency vectors.
- TF-IDF: TF-IDF vectors.
- LSI: LSI vectors with k = 300.
- Doc2Vec: doc2vec vectors with k = 300.

Given a query string, for each embedding print the top-5 most similar entities (entity name, file name, line of code), based on cosine similarity.
