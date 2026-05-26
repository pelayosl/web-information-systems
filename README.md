# 🌐 Web Information Systems
This repository contains small Python projects exploring different concepts of information retrieval on the web. All these projects are part of Universidad de Oviedo's Web Information Systems subject (Sistemas de Información para la Web). The original documents describing the different projects are property of the University and thus cannot be shared, however here I provide small descriptions for each of them.

## 🕷️ Web Crawler
A small web crawler that downloads web pages recursively, grouping them by their domain inside **WARC** files. For this purpose, it uses `.txt` files containing base links to start the crawling.
It includes:
* Configurations for both *depth-first* and *breadth-first* searches.
* Customizable `robots.txt` compliance (ethically, it should always be activated).
* Downloaded WARC files can be opened through the following web page: https://replayweb.page/

## 🧑‍🤝‍🧑 SimHash
This project uses `SimHash` and `SimHashIndex` to detect quasi-duplicate documents inside document collections. Specifically, it uses two collections:
* Chris McCormick's articles collection used for his MinHash project: https://github.com/chrisjmccormick/MinHash/tree/master/data
* A Quora dataset with questions which are potential duplicates (semantically), but expected to behave poorly under lexical analysis techniques

There's a README inside of this specific project explaining a bit better what it does.

>[!WARNING]
The code was developed as a way of learning and understanding SimHash and duplicate detection itself rather than as a useful program. It currently only works for the two collections mentioned, although it provides a baseline for more extensible implementations.

## 📄 Text similarity
This script implements a text similarity search system for a collection of documents and queries, supporting both boolean and frequency-based approaches. It processes a set of documents and queries, converts them into bag-of-words representations (with tokenization, stopword removal, and lemmatization), and then, for each query, finds the most similar document using four different similarity metrics: Dice, Jaccard, Cosine, and Overlapping coefficients.

For documents and queries, the [cranfield dataset](https://www.kaggle.com/datasets/hhhoang/cranfield-dataset) is used.

## 🔎 Lexical Search Engine for LISA collection
This project provides a lexical search engine for processing and retrieving relevant documents from the LISA text corpus. It includes:

* **Indexing**: Tokenizes documents, removes stopwords, applies stemming, and builds an inverted index with TF/IDF weights.
* **Search**: Supports Boolean (AND/OR), cosine similarity, and BM25-based ranking for document retrieval.
* **Utilities**: Functions for loading/saving indices, handling stopwords, and text normalization.

## 🧑‍🤝‍🧑 Semantic Duplicate Detection
This project performs semantic duplicate detection on the Quora question dataset using the [embeddingemma-300m](https://huggingface.co/google/embeddinggemma-300m) for embeddings and ChromaDB. It:
* Loads question embeddings and indexes them in ChromaDB for efficient similarity search
* Uses a transformer-based model to compute semantic similarity between questions
* Identifies duplicate question pairs based on a configurable distance threshold
* Compares detected duplicates to ground truth, reporting precision, recall, and F1 metrics

It requires a HuggingFace token for the embedding model. The project is ideal for experimenting with semantic search and duplicate detection in large text datasets.

## 🎬 Recommender system based on PageRank
This project implements a movie recommender system using the [MovieLens](https://grouplens.org/datasets/movielens/) dataset and the PageRank algorithm. It features:

* Construction of a movie co-occurrence graph based on user ratings
* Personalized PageRank computation for movie recommendations, considering both user preferences and genre similarity (via Jaccard index)
* Generation of recommendation files for specific movies and a global ranking for all movies
* Utilities for text processing and safe file naming

The system allows users to input a movie title and receive top recommendations based on graph-based ranking and genre similarity, making it suitable for experiments in graph-based recommendation and network analysis.

## More coming soon...
