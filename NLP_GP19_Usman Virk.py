"""
@author: Usman Virk (2315254) Group Project 19, Task Owner 5,6,8,10 @ CSE, NLP-Course 2024, Oulu University
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, f1_score, mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from transformers import BertTokenizer, BertModel
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from transformers import RobertaTokenizer, RobertaModel
import torch

# Downloading NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Loading the spaCy model
nlp = spacy.load('en_core_web_lg')


# Loading pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


############################### TASK1 #########################################

# Loading the datasets from Kaggle/food.com
print("Loading dataset...")
df_recipes = pd.read_csv('../521158S-3005_Natural_Language_Processing_and_Text_Mining/Group Project/files/RAW_recipes.csv')
df_interactions = pd.read_csv('../521158S-3005_Natural_Language_Processing_and_Text_Mining/Group Project/files/RAW_interactions.csv')
print("Dataset loaded.")

# Plotting key features: (1) number of ingredients, (2) number of steps, and (3) tags
# Number of ingredients
plt.figure(figsize=(10, 6))
df_recipes['n_ingredients'].hist(bins=30)
plt.title('Distribution of Number of Ingredients')
plt.xlabel('Number of Ingredients')
plt.ylabel('Frequency')
plt.show()

# Number of steps
plt.figure(figsize=(10, 6))
df_recipes['n_steps'].hist(bins=30)
plt.title('Distribution of Number of Steps')
plt.xlabel('Number of Steps')
plt.ylabel('Frequency')
plt.show()

# Tags associated with each recipe
tags = df_recipes['tags'].str.replace('[', '').str.replace(']', '').str.replace("'", '').str.split(',')
tags = tags.explode().str.strip()
plt.figure(figsize=(12, 8))
tags.value_counts().head(20).plot(kind='bar')
plt.title('Top 20 Tags Associated with Recipes')
plt.xlabel('Tags')
plt.ylabel('Frequency')
plt.show()


############################### TASK2 #########################################

# Function to clean and preprocess the text data
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Removing special characters and punctuation
    text = text.lower()  # Converting text to lowercase
    tokens = word_tokenize(text)  # Tokenizing the text
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(tokens)


# Processing data in chunks/rows at a time to avoid memory issues due to limited computational resources
chunk_size = 100  
totallen = 200*chunk_size
df_interactions_subset = df_interactions.head(totallen).copy()
df_recipes_subset = df_recipes.head(totallen).copy()
cleaned_reviews = []
cleaned_description = []
for start in range(0, totallen, chunk_size):
    end = start + chunk_size
    cleaned_reviews_chunk = df_interactions['review'][start:end].apply(lambda x: preprocess_text(str(x)))
    cleaned_description_chunk = df_recipes['description'][start:end].apply(lambda x: preprocess_text(str(x)))

    # Appending the cleaned chunks to the lists
    cleaned_reviews.extend(cleaned_reviews_chunk)
    cleaned_description.extend(cleaned_description_chunk)
    print(f"Processed rows {start} to {end}")

# Assigning the cleaned reviews and descriptions to the DataFrames
df_interactions_subset['cleaned_review'] = cleaned_reviews
df_recipes_subset['cleaned_description'] = cleaned_description
print("Text data preprocessed.")

############################### TASK5 #########################################

# Combining all descriptions and reviews into a single corpus for topic modeling
corpus = df_recipes_subset['cleaned_description'].tolist() + df_interactions_subset['cleaned_review'].tolist()
print("Corpus created.")

# Vectorizing the text data using TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

# Performing LDA for topic modeling
lda_model = LatentDirichletAllocation(n_components=10, random_state=42)
lda_topics = lda_model.fit_transform(tfidf_matrix)

# Function to display topics and their top words
def display_topics(model, feature_names, num_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]))

# Displaying the top 10 words for each topic
num_top_words = 10
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
display_topics(lda_model, tfidf_feature_names, num_top_words)

# Adding topic distribution to the original DataFrames (only for the subset)
df_recipes_subset['topic'] = lda_topics[:len(df_recipes_subset)].argmax(axis=1)
df_interactions_subset['topic'] = lda_topics[len(df_recipes_subset):len(df_recipes_subset) + len(df_interactions_subset)].argmax(axis=1)

# Analyzing how topics correlate with user sentiments and ratings (only for the subset)
topic_sentiment = df_interactions_subset.groupby('topic')['rating'].mean().reset_index()
plt.figure(figsize=(12, 6))
sns.barplot(x='topic', y='rating', data=topic_sentiment)
plt.title('Average Rating by Topic')
plt.xlabel('Topic')
plt.ylabel('Average Rating')
plt.show()

# Displaying the first few rows of the DataFrames to verify the changes (only for the subset)
print(df_recipes_subset[['description', 'cleaned_description', 'topic']].head())
print(df_interactions_subset[['review', 'cleaned_review', 'rating', 'topic']].head())


############################### TASK6 #########################################

food_labels = ['Food', 'PRODUCT', 'NORP', 'INGREDIENT'] 

# Function to extract ingredients and food items using NER
def extract_ingredients(text):
    doc = nlp(text)
    ingredients = [ent.text for ent in doc.ents if ent.label_ in food_labels] 
    print(f"Text: {text}")
    print(f"Extracted Ingredients: {ingredients}")
    return ingredients

# Applying the NER function to a subset of the 'description' column in df_recipes
print("Extracting ingredients from descriptions...")
df_recipes_subset['ner_ingredients'] = df_recipes_subset['cleaned_description'].apply(lambda x: extract_ingredients(str(x)))
print("Ingredients extracted.")

# Displaying the first few rows of the DataFrame to verify the changes (only for the subset)
print(df_recipes_subset[['description', 'ner_ingredients']].head())


# Analyzing how different food content mentions affect user ratings and sentiments
# For simplicity, I assumed that the presence of certain ingredients might influence user ratings

# Merging the interactions and recipes data on recipe_id
df_merged = pd.merge(df_interactions_subset, df_recipes_subset, left_on='recipe_id', right_on='id')


# Extracting ingredients from the merged dataset
df_merged['ingredients'] = df_merged['description'].apply(lambda x: extract_ingredients(str(x)))

# Analyzing the correlation between ingredients and user ratings
ingredient_ratings = df_merged.explode('ingredients').groupby('ingredients')['rating'].mean().reset_index()

# Displaying the top 10 ingredients with the highest average ratings
top_ingredients = ingredient_ratings.sort_values(by='rating', ascending=False).head(10)
print(top_ingredients)

# Visualizing the top 10 ingredients with the highest average ratings
plt.figure(figsize=(12, 6))
sns.barplot(x='rating', y='ingredients', data=top_ingredients)
plt.title('Top 10 Ingredients with Highest Average Ratings')
plt.xlabel('Average Rating')
plt.ylabel('Ingredients')
plt.show()



############################### TASK8 #########################################

# Calculating cosine similarity
similarity_matrix = cosine_similarity(tfidf_matrix)
print(similarity_matrix)

# Function to calculate similarity using Word2Vec embeddings from spaCy
def calculate_similarity_word2vec(text1, text2):
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    return doc1.similarity(doc2)

# I am still working on this part and it's not finalized
# Calculating similarity using Word2Vec
num_recipes = len(df_recipes_subset)
recipe_similarity_word2vec = np.zeros((num_recipes, num_recipes))

for i in range(num_recipes):
    for j in range(num_recipes):
        recipe_similarity_word2vec[i, j] = calculate_similarity_word2vec(
            df_recipes_subset['cleaned_description'].iloc[i],
            df_recipes_subset['cleaned_description'].iloc[j]
        )

print("Word2Vec Similarity Matrix:")
print(recipe_similarity_word2vec)

# Loading pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get BERT embeddings
def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Calculatinging BERT embeddings for all recipes
bert_embeddings = np.array([get_bert_embeddings(desc) for desc in df_recipes_subset['cleaned_description']])

# Calculating cosine similarity using BERT embeddings
similarity_matrix_bert = cosine_similarity(bert_embeddings)
print("BERT Similarity Matrix:")
print(similarity_matrix_bert)


##########################  TASK10 ############################################
# Vectorizing the text data using TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = tfidf_vectorizer.fit_transform(df_interactions_subset['cleaned_review'])
# Defining the target variable (ratings)
y = df_interactions_subset['rating']

# Spliting the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training a Logistic Regression, bnary classification model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)

# Evaluating the model: Calculaing F1 score and confusion matrix
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"F1 Score: {f1}")
print("Confusion Matrix:")
print(conf_matrix)

# Visualizing the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Printing the classification report for all metrics
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)


##########################  TASK11 ############################################

# Calculate average rating for each recipe
average_ratings = df_interactions_subset.groupby('recipe_id')['rating'].mean().reset_index()
average_ratings.columns = ['recipe_id', 'average_rating']  # Rename for clarity

# Check the columns before merging
print("df_recipes columns:", df_recipes.columns)
print("average_ratings columns:", average_ratings.columns)

# Ensure the recipe_id in average_ratings matches the id in df_recipes
average_ratings.rename(columns={'recipe_id': 'id'}, inplace=True)

# Merge average ratings with recipe data
df_recipes_subset = pd.merge(df_recipes_subset, average_ratings, on='id', how='left')

# Check the resulting DataFrame
print(df_recipes_subset.head())

# Vectorizing the text data using TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = tfidf_vectorizer.fit_transform(df_interactions_subset['cleaned_review'])
# Defining the target variable (average ratings)
y = average_ratings['average_rating']

# Spliting the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Training a Logistic Regression model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)

# Evaluating the model: Calculating F1 score and confusion matrix
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"F1 Score: {f1}")
print("Confusion Matrix:")
print(conf_matrix)

# Visualizing the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Printing the classification report for all metrics
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)

# Additional evaluation metrics for regression
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Feature importance analysis (for logistic regression)
coef = model.coef_[0]
feature_names = tfidf_vectorizer.get_feature_names_out()
importance = pd.DataFrame({'Feature': feature_names, 'Importance': coef})
importance = importance.sort_values(by='Importance', ascending=False)

print("Feature Importance:")
print(importance.head(10))

# Report results
print("Average ratings and feature importance analysis completed.")

##########################  TASK12 ############################################
# Function to get embeddings from RoBERTa
def get_roberta_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = roberta_model(inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()  # Mean pooling

# Get RoBERTa embeddings for the cleaned reviews
X_roberta = get_roberta_embeddings(df_interactions_subset['cleaned_review'].tolist())

# Split the data into training and testing sets (80% training, 20% testing)
X_train_roberta, X_test_roberta, y_train, y_test = train_test_split(X_roberta, y, test_size=0.2, random_state=42)

# Training a Logistic Regression model with RoBERTa features
model_roberta = LogisticRegression(max_iter=1000)
model_roberta.fit(X_train_roberta, y_train)

# Making predictions on the test set
y_pred_roberta = model_roberta.predict(X_test_roberta)

# Evaluating the model: Calculating F1 score and confusion matrix
f1_roberta = f1_score(y_test, y_pred_roberta, average='weighted')
conf_matrix_roberta = confusion_matrix(y_test, y_pred_roberta)

print(f"RoBERTa F1 Score: {f1_roberta}")
print("RoBERTa Confusion Matrix:")
print(conf_matrix_roberta)

# Visualizing the confusion matrix for RoBERTa
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_roberta, annot=True, fmt='d', cmap='Blues', xticklabels=model_roberta.classes_, yticklabels=model_roberta.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (RoBERTa)')
plt.show()

# Printing the classification report for RoBERTa
class_report_roberta = classification_report(y_test, y_pred_roberta)
print("RoBERTa Classification Report:")
print(class_report_roberta)

# Additional evaluation metrics for both models
mse_tfidf = mean_squared_error(y_test, y_pred_tfidf)
r2_tfidf = r2_score(y_test, y_pred_tfidf)

mse_roberta = mean_squared_error(y_test, y_pred_roberta)
r2_roberta = r2_score(y_test, y_pred_roberta)

print(f'TF-IDF Mean Squared Error: {mse_tfidf}, R^2 Score: {r2_tfidf}')
print(f'RoBERTa Mean Squared Error: {mse_roberta}, R^2 Score: {r2_roberta}')

# Feature importance analysis for RoBERTa (not directly applicable, but can analyze coefficients)
coef_roberta = model_roberta.coef_[0]
importance_roberta = pd.DataFrame({'Feature': range(len(coef_roberta)), 'Importance': coef_roberta})
importance_roberta = importance_roberta.sort_values(by='Importance', ascending=False)

print("Feature Importance (RoBERTa):")
print(importance_roberta.head(10))

# Report results
print("RoBERTa model training and evaluation completed.")
