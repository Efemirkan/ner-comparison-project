import numpy as np
from collections import Counter

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

    # Store length of sentences
    length = [len(sent) for sent in sents]

    print(f"Median length: {np.median(length)}")
    print(f"95th percentile: {np.percentile(length, 95)}")
    print(f"Max length: {max(length)}")

def check_sanity(sents, tags):

    # Check tags and tokens matched if not print the mismatch
    for s, t in zip(sents, tags):
        assert len(s) == len(t), f"Mismatch in {s} and {t}"

    print(f"Tokens and tags matched")

def tag_distribution(tags):

    # Store all tags in the list
    all_tags = [tag for tags_sent in tags for tag in tags_sent]

    tag_counts = Counter(all_tags)

    # Compute percentage
    total_tokens = sum(tag_counts.values()) # to store number of tokens

    tag_count_per = {} # to store tag and pencentage
    for tag, count in tag_counts.items():
        percentage = round(100 * count / total_tokens, 2) # to compute percentage
        tag_count_per[tag] = percentage

    # Sort by key to compare the sets
    tag_counts = dict(sorted(tag_counts.items()))
    tag_count_per = dict(sorted(tag_count_per.items()))

    return tag_counts, tag_count_per

def entity_distribution(tags):

    # Store all tags in the list
    all_tags = [tag for tags_sent in tags for tag in tags_sent]

    # Store only entities
    entities = []
    for tag in all_tags:

        # Handle with "O"
        if tag == "O":
            entities.append("O")
            continue

        entities.append(tag.split("-")[1])

    entities_counts = Counter(entities)

    without_o_count = {ent: count for ent, count in entities_counts.items() if ent != "O"}

    return entities_counts, without_o_count
