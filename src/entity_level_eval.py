import pandas as pd

# Define extract_entities funtion to extract entities
def extract_entities(tags):

    # INitialize variables
    entities = []
    start = None
    current_type = None

    # Iterate through to extract entities
    for i, tag in enumerate(tags):

        # Check firstly, If 'O' tag and current type is not None otherwise continue next tag
        if tag == 'O':
            if current_type is not None:

                # Append current type, start index, and where tag end to entities
                entities.append((current_type, start, i - 1))

                # Update parameters and continue next tag
                start = None
                current_type = None
            continue

        # If anything else rather than 'O', split the tag    
        prefix, entity_type = tag.split('-', 1)

        # Check if beginning of entity and current_type is not None
        if prefix == 'B':
            if current_type is not None:
                # Append current type, start index, and where tag end to entities
                entities.append((current_type, start, i - 1))

            # Update parameters and continue next tag
            start = i
            current_type = entity_type

        # Check if inside of entity and current_type is not the same entity_type
        elif prefix == 'I':
            if current_type != entity_type:

                if current_type is not None:
                    # Append current type, start index, and where tag end to entities
                    entities.append((current_type, start, i - 1))
                start = i
                current_type = entity_type

    # Check for the last items in the loop and append it
    if current_type is not None:
        entities.append((current_type, start, len(tags) - 1))

    return entities

# Define entity_level_scores funtion to compute evaluation metrics
def entity_level_scores(true_tag_sequences, pred_tag_sequences):
    
    # # INitialize variables to store in required format
    true_entities = []
    pred_entities = []

    # Iterate over to shape in required format
    for sent_id, (true_tags, pred_tags) in enumerate(zip(true_tag_sequences, pred_tag_sequences)):
        true_entities.extend((sent_id, *ent) for ent in extract_entities(true_tags))
        pred_entities.extend((sent_id, *ent) for ent in extract_entities(pred_tags))

    # Create sets to keep only unique values
    true_set = set(true_entities)
    pred_set = set(pred_entities)
    correct = len(true_set & pred_set) # count correctly predicted entities

    # Compute precision, recall and F1 score
    precision = correct / len(pred_set) if pred_set else 0.0
    recall = correct / len(true_set) if true_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    # Return them in a dictionary
    return {'true_entities': len(true_set),
            'predicted_entities': len(pred_set),
            'correct_entities': correct,
            'precision': precision,
            'recall': recall,
            'f1': f1
            }

# Define rebuild_sequences function to splits the flat predictions back into sentence format
def rebuild_sequences(flat_predictions, true_sentences):

    rebuilt = []
    index = 0 # Keep track index for sentence boundiries

    for sent in true_sentences:
        sent_len = len(sent) # Have each sentence length

        # Extract sentences using initial index to end of the sentence
        rebuilt.append(list(flat_predictions[index: index + sent_len]))

        index += sent_len # Update index value

    return rebuilt

def per_entity_span_report(true_tag_sequences, pred_tag_sequences):

    # Initialise lists to store extracted entities
    true_entities = []
    pred_entities = []

    # Iterate over to shape in required format
    for sent_id, (true_tags, pred_tags) in enumerate(zip(true_tag_sequences, pred_tag_sequences)):
        true_entities.extend((sent_id, *ent) for ent in extract_entities(true_tags))
        pred_entities.extend((sent_id, *ent) for ent in extract_entities(pred_tags))

    # Get all unique entity types from both
    entity_types = sorted({ent[1] for ent in true_entities} | {ent[1] for ent in pred_entities})

    rows = []

    # Compute precision, recall and F1 for each entity type
    for entity_type in entity_types:

        # Filter entities by current type
        true_set = {ent for ent in true_entities if ent[1] == entity_type}
        pred_set = {ent for ent in pred_entities if ent[1] == entity_type}
        correct = len(true_set & pred_set) # count correctly predicted entities

        # Compute precision, recall and F1 score
        precision = correct / len(pred_set) if pred_set else 0.0
        recall = correct / len(true_set) if true_set else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        # Append them to rows in a dictionary
        rows.append({
            'entity_type': entity_type,
            'support': len(true_set),
            'predicted': len(pred_set),
            'precision': precision,
            'recall': recall,
            'f1': f1,
        })

    # Return DataFrame
    return pd.DataFrame(rows).sort_values('f1', ascending=False)