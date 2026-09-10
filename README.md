# Tiny Transformer Language Model

A tiny Transformer-based language model built from scratch using PyTorch and deployed with Streamlit.

The goal of this project is not to build a production-grade LLM, but to understand the core components of a Transformer by implementing them manually.

## Project Overview

This project demonstrates the complete flow of a small language model:

```text
Text
 ↓
Tokenization
 ↓
Token IDs
 ↓
Token + Positional Embeddings
 ↓
Multi-Head Self-Attention
 ↓
Feed-Forward Network
 ↓
Residual Connections + LayerNorm
 ↓
Transformer Blocks
 ↓
Language Model Head
 ↓
Next-Token Prediction
 ↓
Autoregressive Generation
```

## What I Built

The Transformer architecture was implemented manually using PyTorch.

Key components include:

* Tokenization
* Vocabulary creation
* Token embeddings
* Positional embeddings
* Query, Key, and Value projections
* Scaled dot-product attention
* Causal masking
* Multi-head attention
* Feed-forward network
* Residual connections
* Layer normalization
* Transformer block
* Stacked Transformer blocks
* Language model head
* Cross-entropy loss
* Backpropagation
* Adam optimizer
* Autoregressive text generation
* `<END>` token for stopping generation

## Training Data

The model was intentionally trained on a very small dataset:

```text
I love cats
I love dogs
Cats are cute
Dogs are friendly
Cats love food
Dogs love food
```

An `<END>` token was added to each sentence during training so the model could learn when a sequence should stop.

## Model Architecture

The final model uses:

```text
Vocabulary Size : 11
Context Length  : 4
Embedding Size  : 4
Attention Heads : 2
FFN Dimension   : 8
Transformer Blocks : 2
```

### Transformer Block

Each Transformer block follows:

```text
Input
  ↓
Multi-Head Self-Attention
  ↓
Residual Connection
  ↓
Layer Normalization
  ↓
Feed-Forward Network
  ↓
Residual Connection
  ↓
Layer Normalization
```

## Training

The model predicts the next token at every position.

For example:

```text
Input:
I love cats <END>

Target:
love cats <END> ...
```

Cross-entropy loss measures how different the predicted token distribution is from the actual next token.

The model parameters are updated using backpropagation and the Adam optimizer.

## Text Generation

The model generates text autoregressively.

For example:

```text
Input:
I love

Output:
I love cats
```

Another example:

```text
Input:
Dogs are

Output:
Dogs are friendly
```

The `<END>` token is used internally to stop generation and is not displayed in the final output.

## Streamlit Application

The trained model was deployed as an interactive Streamlit application.

The application allows users to:

1. Enter a starting phrase
2. Generate text using the trained Transformer
3. Control the maximum number of generated tokens
4. See a graceful message when unknown words are entered

Example:

```text
Input:
Cats are

Output:
Cats are cute
```

## Project Structure

```text
tiny-transformer/
│
├── app.py
├── model.py
├── model.pth
├── vocab.json
├── requirements.txt
└── README.md
```

### Files

`model.py`

Contains the manually implemented Transformer architecture.

`app.py`

Contains the Streamlit user interface and text-generation logic.

`model.pth`

Contains the trained model weights.

`vocab.json`

Contains the token-to-ID and ID-to-token mappings.

`requirements.txt`

Contains the Python dependencies required to run the application.

## Technologies Used

* Python
* PyTorch
* Streamlit
* Google Colab
* GitHub

## Key Learning

Building this project helped me understand that a Transformer is not a single mysterious component.

It is a combination of relatively understandable building blocks:

```text
Embeddings
+
Attention
+
Feed-Forward Networks
+
Residual Connections
+
Layer Normalization
+
Training
+
Next-Token Prediction
```

The model is intentionally tiny and trained on a very small dataset, so it should not be considered a general-purpose language model.

The purpose of this project was to understand the fundamentals by building them from scratch.

## Future Improvements

Possible extensions include:

* Larger training dataset
* Larger embedding dimension
* More Transformer blocks
* Larger vocabulary
* Better tokenizer
* Temperature-based sampling
* Top-k sampling
* Train/validation split
* Model evaluation metrics
* GPU training
* Improved Streamlit interface

## Author

Shailesh Shripathi

Built as a hands-on project to understand Transformer architecture, language modeling, and model deployment.
