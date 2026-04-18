# To extract shape helps the model recognise patterns
def shape_feature(word):
    shape = []

    for ch in word:

        # If uppercase 'X'
        if ch.isupper():
            shape.append('X')

        # If lowercase 'x'
        elif ch.islower():
            shape.append('x')

        # If digit 'd'
        elif ch.isdigit():
            shape.append('d')
            
        else:
            shape.append(ch)

    return ''.join(shape)


def token_to_features(sentence, i):

    token = sentence[i] # to determine token
    word = token['word']
    pos = token['pos']
    chunk = token['chunk']

    # Create features dictionary
    features = {
                'word.lower': word.lower(), # lower form 
                'word[-3:]': word[-3:].lower(), # last 3 char
                'word[-2:]': word[-2:].lower(), # last 2 char
                'word[:3]': word[:3].lower(), # first 3 char
                'word[:2]': word[:2].lower(), # first 2 char
                'word.isupper': word.isupper(), # check if all chars upper
                'word.istitle': word.istitle(), # check if word title form
                'word.isdigit': word.isdigit(), # check if all chars digit
                'word.has_hyphen': '-' in word, # check if has hyphen
                'word.shape': shape_feature(word), # extract the shape
                'pos': pos, # pos tag
                'pos[:2]': pos[:2], # first 2 char of POS tag
                'chunk': chunk, # chunk
                }

    # Create context window
    # If not first word , add previous tokenfeatures
    if i > 0:
        prev_token = sentence[i - 1]
        prev_word = prev_token['word']
        features.update({
            '-1:word.lower': prev_word.lower(),
            '-1:word.istitle': prev_word.istitle(),
            '-1:word.isupper': prev_word.isupper(),
            '-1:pos': prev_token['pos'],
            '-1:chunk': prev_token['chunk'],
        })

    # Else add tag 'BOS' as beginning of sentence
    else:
        features['BOS'] = True

    # If not last word, add next token features
    if i < len(sentence) - 1:
        next_token = sentence[i + 1]
        next_word = next_token['word']
        features.update({
            '+1:word.lower': next_word.lower(),
            '+1:word.istitle': next_word.istitle(),
            '+1:word.isupper': next_word.isupper(),
            '+1:pos': next_token['pos'],
            '+1:chunk': next_token['chunk'],
        })
    
    # Else add tag 'EOS' as ending of sentence
    else:
        features['EOS'] = True

    return features