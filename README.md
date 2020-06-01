# Structured Distributional Model
## Requirements and Usage
The code in this repository is compatible with Python3.x and depends on these libraries:

- pyyaml
- numpy
- neo4j
- scikit-learn
- pandas
- scipy
- tqdm

We recommend to use a virtual environment and to install the specific versions of each library.

To install de pipeline, use the following command under the SDM directory:
```
$ python setup.py install 
```


## Pipeline
The pipeline is divided into two modules: 
- the first module takes as input a (parsed) corpus, processes it and builds a GEK graph
- the second module concerns the SDM computations

### Graph creation

```
$ sdm [-h] [-p {conll,stream}] -i INPUT_DIRS [INPUT_DIRS ...] 
                               -o OUTPUT_DIR
                               --labels LABELS
                               [--delimiter DELIMITER] 
                               [--batch-size-stats BATCH_SIZE_STATS]
                               [--batch-size-events BATCH_SIZE_EVENTS]
                               [--word-thresh WORD_THRESH]
                               [--event-thresh EVENT_THRESH] [-s] [-e]
                               [--workers WORKERS [WORKERS ...]]

optional arguments:
  -h, --help            show this help message and exit
  -p {conll,stream}, --pipeline {conll,stream}
                        type of pipeline to run
  -i INPUT_DIRS [INPUT_DIRS ...], --input_dirs INPUT_DIRS [INPUT_DIRS ...]
                        paths to folder(s) containing corpora
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
  --labels LABELS       path to file for filtering pos/roles
  --delimiter DELIMITER 
  --batch-size-stats BATCH_SIZE_STATS
  --batch-size-events BATCH_SIZE_EVENTS
  --word-thresh WORD_THRESH
  --event-thresh EVENT_THRESH
  -s                    flag to launch lemmas freqs extraction
  -e                    flag to launch events freqs extraction
  --workers WORKERS [WORKERS ...]
```

### SDM
$ sdm extract-possible-relations -U  bolt://127.0.0.1:7687  -u neo4j -p neo4j -o data/results/

$ sdm build-representations -U  bolt://127.0.0.1:7687  -u neo4j -p neo4j -o data/results/ -r data/dataset/generated/dataset.relations -d data/dataset/generated/dataset.all -v path_to_vecs/model.txt

$ sdm compute-evaluation -d data/KS108/original/KS.csv -o data/KS108/results-w2v/ -g data/KS108/results-w2v/KS.svo.out -t ks -v ../word2vecf-SGNS-synf-d300.txt -m data/KS108/original/mapping.csv
***

***
### Credits:
* logger: Alexandre Kabbach
