import numpy as np

def dataset_stats(sents, tags):

    num_sentences = len(sents)  # number of the sentences
    num_tokens = sum(len(token) for token in sents) # number of the tokens
    avg_sentence_length = num_tokens / num_sentences # Average length of the sentences
    num_tags = len(tags) # number of the tags

    print(f"Number of sentences: {num_sentences}")
    print(f"Number of tokens: {num_tokens}")
    print(f"Number of tags: {num_tags}")
    print(f"Average sentence length: {avg_sentence_length}")

def sent_length_distribution(sents):

    length = [len(sent) for sent in sents]

    print(f"Median length: {np.median(length)}")
    print(f"95th percentile: {np.percentile(length, 95)}")
    print(f"Max length: {max(length)}")

def check_sanity(sents, tags):

    for s, t in zip(sents, tags):
        assert len(s) == len(t), f"Mismatch in {s} and {t}"

    print(f"Tokens and tags matched")