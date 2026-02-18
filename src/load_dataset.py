def load_dataset(filepath):

    all_sentences = [] # to store sentences
    current_sentence = [] # to store process sentences


    with open("data/raw/train.txt", 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Handle with sentence boundaries
            if not line:
                if len(current_sentence) > 0:
                    all_sentences.append(current_sentence)
                    current_sentence = [] # to reset the processing sentences
                continue

            # Remove the first line
            if line.startswith("-DOCSTART-"):
                continue
                
            # Split token and tags
            columns = line.split()
            
            if len(columns) >= 4:
                word = columns[0] # Store the words
                tag = columns[-1] # Store the NER label
                
                # Append them in a tuple
                current_sentence.append((word, tag))
                
    # Separate the sentences and the tags
    sentences = [[x for x,y in sentence] for sentence in all_sentences]
    tags = [[y for x,y in sentence] for sentence in all_sentences]

    return sentences, tags