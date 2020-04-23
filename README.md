# Structured Distributional Model

$ sdm extract-possible-relations -U  bolt://127.0.0.1:7687  -u neo4j -p neo4j -o data/results/

$ sdm build-representations -U  bolt://127.0.0.1:7687  -u neo4j -p neo4j -o data/results/ -r data/dataset/generated/dataset.relations -d data/dataset/generated/dataset.all -v path_to_vecs/model.txt


***

***
### Credits:
* logger: Alexandre Kabbach
