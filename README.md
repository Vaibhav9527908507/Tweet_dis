Disaster Tweet Classification Project:-
This project focuses on classifying tweets to identify whether they represent real disasters or not. The implementation involves multiple stages, including data preprocessing, Word2Vec embeddings, handling class imbalances, and using a Random Forest classifier for prediction.

Team Contributions:-

Tejaswini Jadhav-
Step 1: Importing Libraries
Imported essential Python libraries like pandas, nltk, numpy, gensim, and scikit-learn.
Ensured necessary NLTK resources were downloaded for text preprocessing.

Sanjeevani Kadam-
Step 2: Data Loading and Preprocessing
Loaded the dataset and handled missing values using the SimpleImputer with the 'most_frequent' strategy.
Developed a text preprocessing pipeline to clean, tokenize, remove stop words, and lemmatize the tweets.

Rohit Kasar-
Step 3: Word2Vec Embedding with Hyperparameter Tuning
Created a corpus from the preprocessed text for Word2Vec training.
Tuned hyperparameters such as vector_size, window, and min_count to optimize the Word2Vec model.
Built the Word2Vec model with the best parameters.

Smita-
Step 4: Feature Extraction
Extracted document vectors for each tweet using the trained Word2Vec model.
Implemented a function to compute the average of word vectors for tokens present in the Word2Vec vocabulary.

Vaibhav Tayde-
Step 5: Model Training and Evaluation
Addressed class imbalance using SMOTE (Synthetic Minority Oversampling Technique).
Split the resampled data into training and testing sets.
Tuned hyperparameters of the RandomForestClassifier and trained the final model.
Evaluated the model's performance using metrics such as accuracy, precision, recall, and F1-score.

Aappasaheb Khedkar-
Step 6: Example Prediction
Preprocessed new tweets to align with the training pipeline.
Computed document vectors for the new tweet.
Used the trained Random Forest classifier to predict whether a tweet represents a real disaster.

Key Features:-
Data Imputation: Handled missing values in the keyword and location columns.
Text Preprocessing: Cleaned and tokenized tweets, removed stop words, and applied lemmatization.
Word2Vec Embedding: Generated dense vector representations of tweets using the Word2Vec model.
Class Imbalance Handling: Used SMOTE to balance the dataset.
Random Forest Classifier: Tuned and trained a robust classifier for disaster prediction.
