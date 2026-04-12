import pandas as pd

def extract_entity_spans(sentences, tags):
    spans = []

    # Iterate over to match tag and token in token level
    for sent, tag_seq in zip(sentences, tags):

        current_tokens = []
        current_type = None

        for token, tag in zip(sent, tag_seq):

            # If sees O tag append span, then update parameters as empty
            if tag == 'O':
                if current_tokens:
                    spans.append({
                        'entity_type': current_type, # to compare type
                        'span_text': ' '.join(current_tokens), # frequency analysis
                        'span_length': len(current_tokens) # to analyse entity length distribution
                    })
                    current_tokens = []
                    current_type = None
                continue

            prefix, entity_type = tag.split('-', 1) # Split the tag

            # If prefix starts with 'B' or entity type is different, append to span, then update parameters with last token
            if prefix == 'B' or current_type != entity_type:
                if current_tokens:
                    spans.append({
                        'entity_type': current_type,
                        'span_text': ' '.join(current_tokens),
                        'span_length': len(current_tokens)
                    })
                current_tokens = [token]
                current_type = entity_type

            # Otherwise keep in the current token
            else:
                current_tokens.append(token)

        # To save the last span 
        if current_tokens:
            spans.append({
                'entity_type': current_type,
                'span_text': ' '.join(current_tokens),
                'span_length': len(current_tokens)
            })

    # Return as DataFrame
    return pd.DataFrame(spans)