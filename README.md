# Structured Distributional Model
## Requirements and Usage
The code in this repository is compatible with Python3.x and depends on these libraries:

- pyyaml
- numpy
- neo4j
- scikit-learn
- pandas

We recommend to use a virtual environment and to install the specific versions of each library

$ sdm extract-possible-relations -U  bolt://127.0.0.1:7687  -u neo4j -p neo4j -o data/results/

$ sdm build-representations -U  bolt://127.0.0.1:7687  -u neo4j -p neo4j -o data/results/ -r data/dataset/generated/dataset.relations -d data/dataset/generated/dataset.all -v path_to_vecs/model.txt

$ sdm compute-evaluation -d data/KS108/original/KS.csv -o data/KS108/results-w2v/ -g data/KS108/results-w2v/KS.svo.out -t ks -v ../word2vecf-SGNS-synf-d300.txt -m data/KS108/original/mapping.csv
***

***
### Credits:
* logger: Alexandre Kabbach
