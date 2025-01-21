📊 Building a Crowdsourced Recommender System

📝 Project Overview

This project is aimed at developing a recommender system for craft beers using reviews from BeerAdvocate. The goal is to recommend products based on user-defined attributes, leveraging natural language processing (NLP) and data analysis.

👥 Team Members
- Ronak Goyal
- Pranav Garg

📅 Assignment Details

Objective: Build the foundational elements of a crowdsourced recommender system.

🔧 Project Tasks

1. Data Extraction

	•	Source: Top 250 beer reviews from BeerAdvocate.
	•	Goal: Extract approximately 5-6k reviews and filter out non-text reviews to retain around 1700-2000 usable reviews.
	•	Output File Structure:
	•	product_name
	•	product_review
	•	user_rating

2. Attribute Analysis

	•	User Input: Accepts 3 desired product attributes (e.g., “Crisp,” “Robust,” “Fruity”).
	•	Method: Use word frequency analysis to identify key attributes from reviews.
	•	Tip: Perform a lift analysis to verify the co-occurrence of attributes in reviews.

3. Similarity Analysis

	•	Approach: Implement cosine similarity (bag-of-words model).
	•	Output File Structure:
	•	product_name
	•	product_review
	•	similarity_score
	•	Process: Compute similarity scores between user-specified attributes and reviews.

4. Sentiment Analysis

	•	Tool: Use VADER (or another NLP model).
	•	Customization: Modify the default VADER lexicon if necessary for contextual accuracy.
	•	Goal: Assign sentiment scores to each review.

5. Evaluation Score

	•	Calculation: Combine similarity and sentiment scores to generate an overall evaluation score.
	•	Objective: Use this combined score to recommend the top 3 products.

6. Word Vector Comparison

	•	Tool: Use word vectors (e.g., spaCy medium-sized pretrained vectors) and compare results with the bag-of-words approach.
	•	Analysis: Evaluate if word embeddings improve recommendations and check attribute mentions across reviews.

7. Alternative Recommendations

	•	Analysis: Compare the evaluation score recommendations with the top 3 highest-rated products.
	•	Justification: Determine if highly-rated products meet user-specified attributes.

8. Product Similarity Analysis

	•	Task: Choose 10 beers from the dataset and identify the most similar beer to one of them.
	•	Method: Explain the logic and methodology used.

🛠️ Installation and Setup

Required Libraries

Ensure you have the required libraries installed:

pip install selenium spacy nltk pandas numpy matplotlib scikit-learn
!python -m spacy download en_core_web_sm
!python -m spacy download en_core_web_md

Key Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from selenium import webdriver

🚀 How to Run

Step-by-Step Guide

	1.	Clone the Repository
Open your terminal or command prompt and run:

git clone https://github.com/pranvgarg/Building-a-Crowdsourced-Recommender-System.git
cd Building-a-Crowdsourced-Recommender-System


	2.	Set Up the Python Environment
Create and activate a virtual environment:

python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate


	3.	Install Dependencies
Run the following command:

pip install -r requirements.txt

	If requirements.txt is unavailable, manually install dependencies:

pip install pandas numpy matplotlib nltk spacy selenium scikit-learn
!python -m spacy download en_core_web_sm
!python -m spacy download en_core_web_md


	4.	Run the Python Notebook
Start Jupyter Notebook:

jupyter notebook

Open main_notebook.ipynb and execute the cells sequentially.

	5.	Input Product Attributes
Locate the user input cell in the notebook:
	•	Enter three attributes (e.g., “Balanced,” “Fruity,” “Robust”).
	•	Run the cell to proceed with similarity and sentiment analysis.
	6.	Generate Recommendations
	•	Execute cells to compute similarity and sentiment scores.
	•	View the top 3 recommended products based on the evaluation score.
	7.	Analyze Outputs
	•	Review generated tables showing product_name, product_review, similarity_score, and sentiment_score.
	•	Check visualizations for better insights into recommendation results.
	8.	Optional: Use Word Vectors
	•	Run the cell that evaluates word vectors (e.g., spaCy) for an alternative recommendation approach.
	•	Compare results to the bag-of-words model to assess any differences.
	9.	Export Results
Save the final recommendations to a CSV file or review them directly in the notebook.

📈 Analysis & Insights

	•	Bag-of-Words vs Word Vectors: Highlight differences and attribute coverage across approaches.
	•	Top-Rated vs Evaluated Products: Analyze how recommendations align with user-specified needs.


📚 Future Work

	•	Broaden to other product types.
	•	Integrate more advanced NLP models.
