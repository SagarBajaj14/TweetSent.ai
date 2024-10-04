# **TweetSent.ai: Twitter Sentiment Analysis Using ML, DL, & Transformer Models**

TweetSent.ai is a comprehensive project focused on Twitter sentiment analysis using a wide array of machine learning, deep learning, and transformer architectures. This project explores sentiment analysis through traditional machine learnign models such as Bernoulli Naive Bayes, SVM, Logistic Regression, and Random Forest using TF-IDF feature vectors. Additionally, deep learning models including CNN and RNN with Word-Index, GloVe, GloVe Twitter, and FastText embeddings are explored, alongside fine-tuning transformer models such as BERT, DistilBERT, BERTweet, and RoBERTa for the task of sentiment analysis. The Sentiment140 Dataset with 1.6 million tweets is used for the task of twitter sentiment analysis. 

## Table of Contents
- [Overview](#overview)
- [Key Objectives](#key-objectives)
- [Project Structure](#project-structure)
- [Features](#features)
- [Models Used](#models-used)
- [Results](#results)


## Overview
Social media platforms like Twitter are brimming with user-generated data. **TweetSent AI** aims to classify the sentiment of tweets (positive or negative) using various machine learning and deep learning models. The project implements both classical models and state-of-the-art transformers to provide a robust analysis pipeline.

### Key Objectives:
- Preprocess Twitter data for sentiment analysis.
- Train and evaluate traditional machine learning models using TF-IDF feature vectors.
- Implement deep learning models like CNN and RNN with different embeddings.
- Integrate advanced Transformer models with their pre-trained embeddings.
- Provide an interactive platform to visualize and compare model performance.

## Project Structure

```bash
TweetSent AI/
│
├── data/               # Dataset and preprocessing scripts
├── model/              # Pretrained model instances for inference
├── code/               # Jupyter Notebooks for analysis and experiments
├── reference papers/   # Research Papers for understanding
├── Detailed review doc/   # Final doc summarizing the project
├── README.md           # Project readme file
└── streamlit-dash/     # Streamlit app for real-time sentiment analysis

## Features

### Machine Learning Models:
- **Bernoulli Naive Bayes**
- **SVM**
- **Logistic Regression**
- **Random Forest**
  - Trained on TF-IDF feature vectors

### Deep Learning Models:
- **CNN** and **RNN** models trained with:
  - Word-Index embeddings
  - GloVe embeddings
  - GloVe Twitter embeddings
  - FastText embeddings

### Transformer Models:
- **BERT**
- **DistilBERT**
- **BERTweet**
- **RoBERTa**
  - Each with their own tokenizers and embeddings

### Data Preprocessing:
- Text cleaning: remove URLs, hashtags, mentions, emojis, etc.
- Tokenization, lemmatization, and custom handling of emojis and hashtags

### Evaluation Metrics:
- Accuracy
- F1-Score
- Precision
- Recall
- Confusion Matrices

### Interactive Visualization:
- Real-time tweet sentiment prediction using a Streamlit app

## Models Used

### Machine Learning Models
- **Bernoulli Naive Bayes**: Trained with TF-IDF vectors.
- **Support Vector Machine (SVM)**: Utilizes hyperplanes for classification.
- **Logistic Regression**: Linear model for binary classification.
- **Random Forest**: An ensemble of decision trees for robust classification.

### Deep Learning Models
- **CNN**: Convolutional Neural Networks with Word-Index, GloVe, GloVe Twitter, and FastText embeddings.
- **RNN**: Recurrent Neural Networks with various embeddings for sequential data analysis.

### Transformer Models
- **BERT**: Bidirectional encoder with pre-trained embeddings.
- **DistilBERT**: Lightweight version of BERT, faster with minimal loss in accuracy.
- **BERTweet**: Transformer pre-trained on Twitter data.
- **RoBERTa**: A robust version of BERT optimized for better performance.

## Results
Key findings include:

- **Best Machine Learning Model**: Logistic Regressor achieved the highest accuracy with TF-IDF vectors.
- **Best Deep Learning Model**: RNN with GloVe Twitter embeddings outperformed others in terms of F1-Score.
- **Best Transformer Model**: BERTweet achieved the best overall accuracy and F1-Score on the test set.



