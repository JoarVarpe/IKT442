from collections import OrderedDict
import nlp
import numpy as np
import spacy
import networkx as nx
from evaluate_rouge import make_sentences_into_summary, total_rouge, make_random_comparison
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from spacy.lang.en.stop_words import STOP_WORDS
from datasets import load_dataset
import tqdm
from transformers import BertTokenizer, BertModel
import torch


def make_spacy_gen_indexes_into_list(spacy):
    actual_list = []
    for sent in spacy:
        actual_list.append(sent)
    return actual_list


tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
model = BertModel.from_pretrained("bert-base-multilingual-uncased")

nlp_nb = spacy.load('en_core_web_lg')
test_dataset = load_dataset("cnn_dailymail", "3.0.0", split="test")
articles = test_dataset["article"]
highlights = test_dataset["highlights"]
rouge = nlp.load_metric('rouge')

list_of_highlights = []
list_of_summaries = []
list_of_random_summaries = []

for iteration, t in tqdm.tqdm(enumerate(zip(articles, highlights))):

    doc = nlp_nb(t[0])

    list_of_sents = []
    for sent in doc.sents:
        encoded_input = tokenizer(sent.text, max_length=512, return_tensors='pt')
        with torch.no_grad():
            list_of_sents.append(model(**encoded_input, return_dict=True)["pooler_output"].squeeze(0).numpy())

    enc_sents = np.vstack(list_of_sents)

    dist = cosine_similarity(enc_sents)
    np.fill_diagonal(dist, 0)
    dist1 = dist.copy()
    dist.argmax(axis=1)

    similarity_and_tuples = []
    checked_similarity = []
    largest_sim = 0
    for i in range(len(dist1)):
        for k in range(len(dist1)):
            if k == i:
                continue
            if (i, k) in checked_similarity or (k, i) in checked_similarity:
                continue
            if dist[i][k] > largest_sim:
                largest_sim = dist[i][k]
            # print(dist[i][k], i, k)
            checked_similarity.append((i, k))
            similarity_and_tuples.append((i, k, dist[i][k]))

    G = nx.Graph()
    # make tuples into nodes
    # print(largest_sim)
    threshold = 0.6

    for tuple in similarity_and_tuples:
        if threshold * largest_sim < tuple[2]:
            src_node = tuple[0]
            dest_node = tuple[1]
            weight = tuple[2]
            G.add_edge(src_node, dest_node, weight=weight)

    pos = nx.spring_layout(G)  # compute graph layout
    nx.draw(G, pos, node_size=700)  # draw nodes and edges
    nx.draw_networkx_labels(G, pos)  # draw node labels/names
    # draw edge weights
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    # show image
    # plt.show()
    pr = nx.pagerank(G, max_iter=1000, alpha=0.9)

    max_key = max(pr, key=pr.get)
    from operator import itemgetter

    list_of_result_tuples = sorted(pr.items(), key=itemgetter(1), reverse=True)
    list_of_order = sorted(pr, key=pr.get, reverse=True)

    list_of_sentences = make_spacy_gen_indexes_into_list(doc.sents)

    # put together sentences by order of importance
    list_of_summaries.append(make_sentences_into_summary(list_of_result_tuples, list_of_sentences))
    list_of_random_summaries.append(make_random_comparison(list_of_sentences))
    list_of_highlights.append(t[1])
    if iteration == 200:
        break

print("actual rouge:")
total_rouge(list_of_summaries, list_of_highlights, rouge)
print("random rouge:")
rouge_rand = nlp.load_metric('rouge')
total_rouge(list_of_random_summaries, list_of_highlights, rouge_rand)
