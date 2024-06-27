__authors__ = 'Raquel G. Alhama and Nina Delcaro'

import os, sys, inspect
import random
import torch
import numpy as np
from collections import Counter
from s2i import String2IntegerMapper


def init_seed(seed):

    #set the seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False

def sentences2indexs(sentences, beginning, ending, word_mappings):
    """
        Creates the dictionary of mappings and returns the mapped sentences.
        sentences: list of list of strings
        separator: string to separate sentences
    """
    
    all_sentences_indexed = []
    for _, sentence in enumerate(sentences):
        sentence_idxs = []
        #words = sentence.split(" ")
        sentence.insert(0, beginning)
        sentence.append(ending)
        for word in sentence:
            #Remove spaces at beginning or ending of words
            #word=word.strip()
            word_mappings.add_string(word)          # Add word to mappings
            idx = word_mappings[word]               # Get index of the word
            sentence_idxs.append(idx)
        all_sentences_indexed.append(sentence_idxs)

    return all_sentences_indexed

def prepare_sequences(all_sentences_indexed):
    input_seq, target_seq = [], []
    # Prepare input and target idx sequences (i.e remove last and first items, respectively)
    for i in range(len(all_sentences_indexed)):
        # Remove last item for input sequence (we don't need to predict anything after this one)
        input_seq.append(all_sentences_indexed[i][:-1])

        # Remove first item for target sequence
        target_seq.append(all_sentences_indexed[i][1:])


    return input_seq, target_seq


def prepare_tensors(inputs, targets, device):
    """
    Prepare data for training of minibatch of size 1 ('SGD')
    We don't need padding/packing.
    """
    inputs_t, targets_t = [], []
    for input, target in zip(inputs, targets):
        # Clone and detach tensors instead of recreating them
        input_t = torch.as_tensor(input, dtype=torch.long, device=device).clone().detach()
        target_t = torch.as_tensor(target, dtype=torch.long, device=device).clone().detach()
        inputs_t.append(input_t)
        targets_t.append(target_t)
    return inputs_t, targets_t


def rare_words_to_unknown(sentences, min_frequency):
    
    # flatten the sentences list
    sentences_flat = [word for sent in sentences for word in sent]
    
    # create a word frequency dictionary
    freq_dict = Counter(sentences_flat)
    
    # save only words that appear more than min_frequency times
    frequent_enough_words = {word:freq for (word,freq) in freq_dict.items() if freq >= min_frequency}
    
    # change the rare words in sentences to UNK
    processed_sentences = [[word if word in frequent_enough_words else '<UNK>' for word in sent]for sent in sentences]

    return processed_sentences


def input_to_indexs(sentences, min_frequency):


    # Create mapping dictionary
    word_mappings = String2IntegerMapper()

    # Define null, and beginning and ending sentence symbol
    beginning = "<s>"
    ending = "</s>"
    null = "<NULL>"

    #Add to dictionary
    word_mappings.add_string(null) #we reserve 0 for null)
    word_mappings.add_string(beginning)
    word_mappings.add_string(ending)

    processed_sentences = rare_words_to_unknown(sentences, min_frequency)
    
    # Convert sentences into word indexes
    all_sentences_indexed = sentences2indexs(processed_sentences, beginning, ending, word_mappings)

    return word_mappings, all_sentences_indexed


def create_batches(inputs, targets, batch_size, pad_idx=0):
    """
    Take a list of input sequences and their corresponding target sequences, then group them into batches
    for training a model. Each batch has fixed number of sentences. 

    pad_idx: fills out all shorter sentences in a batch to match the length of the longest sentence in te batch.
    """
    # Calculate the number of batches needed to process all data given the specified batch size
    num_batches = len(inputs) // batch_size + (len(inputs) % batch_size != 0)

    # Initialize lists to store batched inputs and targets
    batched_inputs, batched_targets = [], []

    # Iterate through each batch
    for i in range(num_batches):
        # Calculate the start and end indices for this batch
        start = i * batch_size
        end = start + batch_size

        # Slice the inputs and targets to create a batch from start index to end index
        batch_inputs = inputs[start:end]
        batch_targets = targets[start:end]

        # Determine the maximum length of the sequences in this batch to pad all sequences to this length
        max_length = max(len(input) for input in batch_inputs)

        # Pad each sequence in the batch so they all have the same length (max_length)
        padded_inputs = [input + [pad_idx] * (max_length - len(input)) for input in batch_inputs]
        padded_targets = [target + [pad_idx] * (max_length - len(target)) for target in batch_targets]

        # Convert the list of padded inputs and targets to PyTorch tensors and append to the lists
        batched_inputs.append(torch.tensor(padded_inputs, dtype=torch.long))
        batched_targets.append(torch.tensor(padded_targets, dtype=torch.long))

    return batched_inputs, batched_targets