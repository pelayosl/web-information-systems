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

