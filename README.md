# Sentiment Analysis API

## Overview

This project provides a backend API for sentiment analysis, designed to classify reviews into positive, neutral, and negative sentiments. It also includes additional features such as keyword extraction, word cloud generation, and summary interpretation.

## Features

- **Sentiment Classification**:  
  A pretrained model classifies sentiments as Positive, Neutral, and Negative.
  
- **Keyword Extraction**:  
  Identifies key phrases and terms from the review data.
  
- **Visualizations**:  
  Generates word clouds for quick identification of common terms.
  
- **Summaries**:  
  Uses an LLM to provide concise summaries of key themes and issues.

## Model Training

- **Dataset**:  
  Amazon fashion review data: [https://www.kaggle.com/datasets/haoboxu/amazon-reviews-for-sentiment-analysis](https://www.kaggle.com/datasets/haoboxu/amazon-reviews-for-sentiment-analysis)

- **Performance Metrics**:  
  - Precision: 73.2%  
  - Recall: 73.2%

## Getting Started

### Prerequisites

- Python 3.8+
- FastAPI
- Pydantic
- FastText
- spaCy
- pandas
- scikit-learn
- wordcloud
- matplotlib

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/sentiment-analysis-api.git
   cd sentiment-analysis-api
