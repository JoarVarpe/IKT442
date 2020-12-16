import nlp
import random


def total_rouge(list_of_summaries, list_of_highlights, rouge):
    for lp, lg in zip(list_of_summaries, list_of_highlights):
        rouge.add(lp, lg)
    score = rouge.compute()
    for k, s in score.items():
        print(k, s.mid.fmeasure)
    print("=======================================")


def make_random_comparison(list_of_sentences, num_sent=3):
    sentences = []
    the_chosen_sentences = random.sample(range(len(list_of_sentences)), num_sent)
    for index in the_chosen_sentences:
        sentences.append(list_of_sentences[index].text)

    for i, sentence in enumerate(sentences):
        sub_list = sentence.split("\n")
        new_sentence = "".join(sub_list)
        sentences[i] = new_sentence

    the_result = "".join(sentences)

    return the_result


def make_sentences_into_summary(list_of_result_tuples, list_of_sentences, num_sent=3):
    total = []
    sentences = []
    for i in range(num_sent):
        total.append(list_of_result_tuples[i])

    for i in range(len(sorted(total))):
        sentences.append(list_of_sentences[sorted(total)[i][0]].text)

    for i, sentence in enumerate(sentences):
        sub_list = sentence.split("\n")
        new_sentence = "".join(sub_list)
        sentences[i] = new_sentence

    the_result = "".join(sentences)

    return the_result
