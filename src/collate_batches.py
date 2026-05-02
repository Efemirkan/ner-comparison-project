import torch

def collate_batches(batch, tag_to_id):

    batch_size = len(batch) # To handle dynamic batch size

    
    # Find padding limits for sentences and words
    max_sent_len = max(len(item['word_ids']) for item in batch)
    max_word_len = max(max(len(chars) for chars in item['char_ids']) for item in batch)

    # Create empty boxes as template set max sentence length, pre-padding 
    word_tensor = torch.zeros(batch_size, max_sent_len, dtype=torch.long) # token level
    char_tensor = torch.zeros(batch_size, max_sent_len, max_word_len, dtype=torch.long) # char level
    tag_tensor = torch.full((batch_size, max_sent_len), tag_to_id['O'], dtype=torch.long) # tags, filled with 'O' as default

    mask = torch.zeros(batch_size, max_sent_len, dtype=torch.bool) # masking tensor
    words = [] # To store original words

    # Fill tensors with real data, iterate over for index of sentence in batch
    for i, item in enumerate(batch):

        sent_len = len(item['word_ids']) # actual length of sentence
        words.append(item['words']) # Append to 'words' list, original word

        # Stacking word IDs 2D
        word_tensor[i, :sent_len] = torch.tensor(item['word_ids'], dtype=torch.long)

        # Stacking tag IDs 2D
        tag_tensor[i, :sent_len] = torch.tensor(item['tag_ids'], dtype=torch.long)

        # mark real tokens as True in mask
        mask[i, :sent_len] = True

        # Stacking character IDs 3D
        for j, chars in enumerate(item['char_ids']):
            char_tensor[i, j, :len(chars)] = torch.tensor(chars, dtype=torch.long)


    return {'words': words, # word
            'word_ids': word_tensor, # word id, input for word embedding
            'char_ids': char_tensor, # char id, input for char embedding
            'tag_ids': tag_tensor, # tag id
            'mask': mask # tells model to ignore the zeros during loss calculation and attention
            }