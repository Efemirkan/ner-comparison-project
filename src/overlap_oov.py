import pandas as pd

# Define fuction to create token level list
def token_level(sentences):
    return [token for sent in sentences for token in sent]

# Define function to inspect oov words
def token_oov_df(sentences, tags, train_vocab, split_name):

    rows = []

    # Iterate over to match token and tag
    for sent, tag_seq in zip(sentences, tags):
        for token, tag in zip(sent, tag_seq):

            entity_type = tag.split('-')[-1] if tag != 'O' else 'O' # split the entity if not O

            rows.append({
                        'token': token,
                        'tag': tag,
                        'entity_type': entity_type,
                        'is_oov': int(token not in train_vocab),
                        'split': split_name
                    })
            
    return pd.DataFrame(rows)