import torch
import numpy as np
PAD_TOKEN = '<PAD>'

def load_dataset(filepath):

    all_sentences = [] # to store sentences
    current_sentence = [] # to store process sentences

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Handle with sentence boundaries
            if not line:
                if current_sentence:
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

def load_dataset_lr(filepath):

    all_sentences = [] # to store sentences
    current_sentence = [] # to store process sentences

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Handle with sentence boundaries
            if not line:
                if current_sentence:
                    all_sentences.append(current_sentence)
                    current_sentence = []
                continue
            
            # Remove the first line
            if line.startswith('-DOCSTART-'):
                continue

            # Split token and tags
            parts = line.split()

            # Skip if any column missing or extra
            if len(parts) != 4:
                continue
            
            # Unpack parts 
            word, pos, chunk, ner = parts

            # Append to current sentences in require form
            current_sentence.append({
                                    'word': word,
                                    'pos': pos,
                                    'chunk': chunk,
                                    'ner': ner,
                                    })

    # To handle with last sentence, but actually we do not need for this dataset
    if current_sentence:
        all_sentences.append(current_sentence)

    return all_sentences

def load_glove(path, word_to_id):

    # Create empty box for embedding
    embedding_matrix = np.random.normal(scale=0.5, size=(len(word_to_id), 100))
    embedding_matrix[word_to_id[PAD_TOKEN]] = np.zeros(100) # Initialize pad tokens to zeros same embedding dimension

    count = 0 # to count matched words

    # Read embedding file
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:

            # Split and clean the line
            values = line.rstrip().split()
            word = values[0] # Extract the word

            # Check if word in vocabulary
            if word in word_to_id:

                # Find the word index
                idx = word_to_id[word]

                # Create a vector from word embedding values from GloVe data
                vector = np.array(values[1:])

                # Check if vector dimension equal embedding dimension
                if vector.shape[0] == 100:

                    # Assign index to vector 
                    embedding_matrix[idx] = vector 
                    count += 1 # Add to count

    # Return embedding matrix type float32 and count
    return torch.tensor(embedding_matrix, dtype=torch.float32), count