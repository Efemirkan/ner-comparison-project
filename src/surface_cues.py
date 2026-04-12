import pandas as pd

def token_shape_features(token):
    return {"is_capitalised": int(token[:1].isupper()),
            "is_all_caps": int(token.isupper() and any(ch.isalpha() for ch in token)),
            "has_digit": int(any(ch.isdigit() for ch in token)),
            "has_hyphen": int('-' in token),
            "is_title_case": int(token.istitle()),
            "has_punctuation": int(any(not ch.isalnum() for ch in token)),
            "has_suffix": token[-3:].lower() if len(token) >= 3 else token.lower()
            }

def build_surface_df(sentences, tags):

    rows = []
    
    # Iterate over to match tag-token in token level
    for sent, tag_seq in zip(sentences, tags):
        for token, tag in zip(sent, tag_seq):

            features = token_shape_features(token)
            entity_type = tag.split('-')[-1] if tag != 'O' else 'O' # Pick up entity type only

            rows.append({"token": token, 
                         "tag": tag, 
                         "entity_type": entity_type, 
                         "is_entity": int(tag != 'O'), 
                         **features # Unpack all key-value pairs
                         })
            
    return pd.DataFrame(rows)