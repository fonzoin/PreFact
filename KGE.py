from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
import pandas as pd
import torch
import os
import datetime

"""convert kg file for KGE"""
def convert_kg_file(input_file, output_file, num_relations):
    with open(input_file, 'r', encoding='utf-8') as infile:
        content = infile.readlines()

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for line in content:
            new_line = '\t'.join(line.strip().split())
            outfile.write(new_line + '\n')
        for line in content:
            new_line = line.strip().split()
            new_line[1] = str(int(new_line[1]) + num_relations)
            reversed_line = '\t'.join(reversed(new_line))
            outfile.write(reversed_line + '\n')


dataset = 'alibaba-ifashion'
input_file = os.path.join("data", dataset, "kg_final.txt")
output_file = os.path.join("data", dataset, "kg_final.tsv")
df = pd.read_csv(input_file, sep=' ', header=None)
num_entities = max(max(df[0]), max(df[2])) + 1
num_relations = max(df[1]) + 1
convert_kg_file(input_file, output_file, num_relations)
entity_to_id = {str(i): i for i in range(num_entities)}
relation_to_id = {str(i): i for i in range(2 * num_relations)}
model_name = 'RotatE'
KGE_results_file = os.path.join("pretrained", f"{model_name}_{dataset}")

tf = TriplesFactory.from_path(output_file, create_inverse_triples=False, entity_to_id=entity_to_id, relation_to_id=relation_to_id)
training, testing, validation = tf.split([.8, .1, .1])
result = pipeline(
    training=training,
    testing=testing,
    validation=validation,
    model=model_name,
    model_kwargs={
        'embedding_dim': 32
    },
    stopper='early',
    epochs=200
)
result.save_to_directory(KGE_results_file)
