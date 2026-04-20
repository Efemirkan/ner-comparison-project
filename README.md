# ner-comparison-project

# Feature-Based vs Neural Sequence Models for Named Entity Recognition(NER)

## Overview

This project compares two different approaches to NER.
- A classical "feature-based logistic regression" model
- A BiLSTM neural sequence model

The main goal is to analyse how these models behave under data sparsity, when training data is reduced.

The project investigates:
- How modelling assumptions affect performance
- How models handle rare entities
- Whether sequential neural models generalise better

## Dataset

The experiments use the CoNLL-2003 NER dataset.

## Models

1- Feature-Based Baseline
Model: Logistic Regression

Features include:
- The current word
- Lowercase form
- Capitalisation
- Common Prefixes and Suffixes
- PoS tag
- Chunk tag
- Simple token shape cues digits, hyphens and punctuation
- A small context window around the token
- Sentence boundary markers

The key assumption of this model is that each token is classified independently, meaning it does not explicitly model dependencies between neighbouring labels.

2- Neural Sequence Model
Model: Bidirectional LSTM

Architecture:

- Embedding layer
- BiLSTM
- Linear layer
- Softmax output per token

The BiLSTM captures both left and right context, which is important for entity boundaries.

## Data Sparsity Experiments

To simulate low-resource settings, both models are trained on:

- 100% of training data
- 50%
- 25%
- 10%

to determine which model degrades more gracefully under sparsity.

## Evaluation

The models are evaluated using:

- Token-level accuracy
- Token-level precision, recall, and F1
- Token-level macro and weighted F1
- Entity-level precision, recall, and F1
- Per-entity span-level F1

In addition to overall performance, I also analyse:

- Confusion matrix for non-O labels

This provides both token-level and entity-level perspectives, allowing me to evaluate not only how well labels are predicted individually, but also how well complete entity spans are identified.

For the full comparison between models, I further consider more detailed error patterns, including:

- Rare entities (less than 5 occurrences)
- Frequent entities (more than 20 occurrences)
- Boundary detection errors
- Confusion between similar classes

These analyses are used to better understand model behaviour under data sparsity and to explain performance differences between the feature-based and neural approaches.

## Why This Project

This project will demonstrate:

- Understanding of classical NLP methods
- Implementation of neural sequence models in PyTorch
- Experimental design under controlled data conditions
- Analytical evaluation beyond accuracy

## Repository Structure

```
ner-comparison-project/
│
├── data/
├── notebooks/
│   ├── 00_data_exploration.ipynb
│   ├── 01_feature_based_model.ipynb
│   ├── 02_bilstm_model.ipynb
│   ├── 03_sparsity_experiments.ipynb
│
├── models/
├── src/
├── results/
└── README.md
```


## Author

Efe Mirkan Guner </br>
MSc Artificial Intelligence & Adaptive Systems</br>
University of Sussex


  
