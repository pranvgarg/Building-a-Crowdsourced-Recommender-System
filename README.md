📊 Building a Crowdsourced Recommender System

📝 Project Overview

This project is part of the MSBA F2024 course assignment, aimed at developing a recommender system for craft beers using reviews from BeerAdvocate. The goal is to recommend products based on user-defined attributes, leveraging natural language processing (NLP) and data analysis.

📅 Project Details

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

!pip install selenium spacy nltk
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

Preprocessing Steps

	1.	Web Scraping: Use Selenium to extract reviews.
	2.	Text Cleaning: Tokenize and clean review data using NLTK and Python string manipulation.
	3.	Vectorization: Apply TF-IDF vectorization for similarity analysis.
	4.	NLP Model: Use spaCy for advanced NLP tasks.

🏆 Key Outputs

	•	Data Files:
	•	beer_reviews.csv containing product_name, product_review, and user_rating.
	•	analysis_output.csv with product_name, product_review, similarity_score, and sentiment_score.
	•	Tables and Visualizations: Display top recommendations, analysis charts, and comparison results.

🔍 Example Code Snippets

Cosine Similarity Calculation

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(reviews)
cosine_sim = cosine_similarity(tfidf_matrix, user_input_vector)

Sentiment Analysis

sid = SentimentIntensityAnalyzer()
review['sentiment_score'] = review['product_review'].apply(lambda x: sid.polarity_scores(x)['compound'])

📈 Analysis & Insights

	•	Bag-of-Words vs Word Vectors: Discuss differences in recommendations and attribute coverage.
	•	Top Rated Products: Evaluate if top-rated beers align with user-specified preferences.

🎨 Visuals

Include relevant tables, plots, or images here for clarity.

🚀 How to Run

	1.	Clone the repository.
	2.	Run the Python notebook.
	3.	Input desired product attributes.
	4.	Generate and view recommendations.

📚 Future Work

	•	Expand to other product categories.
	•	Integrate more sophisticated NLP models.

Ensure you upload any visual results to a GitHub images folder and reference them using Markdown image syntax, e.g., ![alt text](images/sample-chart.png) for seamless GitHub integration.
