from collections import OrderedDict
import nlp
import os
import numpy as np
import spacy
import networkx as nx
from evaluate_rouge import make_sentences_into_summary, total_rouge, make_random_comparison
import matplotlib.pyplot as plt
from spacy.lang.nb.stop_words import STOP_WORDS

rouge = nlp.load_metric('rouge')
path = "FILE_PATH"
text_dict = {}
predone_summary_dict = {}

list_of_highlights = []
list_of_summaries = []

list_of_random_summaries = []


def make_spacy_gen_indexes_into_list(spacy):
    actual_list = []
    for sent in spacy:
        actual_list.append(sent)
    return actual_list


for k, file in enumerate(os.listdir(path)):
    filename = os.path.join(path, file)
    with open(filename, "r") as myfile:
        text = myfile.read()
    new_text = text.split("===")
    predone_summary_dict[k] = new_text[0]
    text_dict[k] = new_text[1]

for key, v in text_dict.items():

    nlp_nb = spacy.load('nb_core_news_lg')
    doc = nlp_nb(text_dict[key])
    list_of_sentences = make_spacy_gen_indexes_into_list(doc.sents)

    similarity_and_tuples = []
    checked_similarity = []
    largest_sim = 0
    for i in range(len(list_of_sentences)):
        for k in range(len(list_of_sentences)):
            if k == i:
                continue

            if (i, k) in checked_similarity or (k, i) in checked_similarity:
                continue
            if list_of_sentences[i].similarity(list_of_sentences[k]) > largest_sim:
                largest_sim = list_of_sentences[i].similarity(list_of_sentences[k])

            checked_similarity.append((i, k))
            similarity_and_tuples.append((i, k, list_of_sentences[i].similarity(list_of_sentences[k])))

    dict_of_tuples = {}

    for i in range(len(list_of_sentences)):
        dict_of_tuples[i] = []
        for tuple in similarity_and_tuples:
            if (i == tuple[0] or i == tuple[1]):
                dict_of_tuples[i].append(tuple)

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
    plt.show()
    pr = nx.pagerank(G, max_iter=1000, alpha=0.9)
    plt.show()
    max_key = max(pr, key=pr.get)
    from operator import itemgetter

    list_of_result_tuples = sorted(pr.items(), key=itemgetter(1), reverse=True)
    list_of_order = sorted(pr, key=pr.get, reverse=True)

    # put together sentences by order of importance
    list_of_summaries.append(make_sentences_into_summary(list_of_result_tuples, list_of_sentences))
    list_of_random_summaries.append(make_random_comparison(list_of_sentences))
    list_of_highlights.append(predone_summary_dict[key])

print("actual rouge:")
total_rouge(list_of_summaries, list_of_highlights, rouge)
print("random rouge")
rouge_rand = nlp.load_metric('rouge')
total_rouge(list_of_random_summaries, list_of_highlights, rouge_rand)

print("DONE")
