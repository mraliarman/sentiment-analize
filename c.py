import pandas as pd
from sklearn.model_selection import train_test_split
from textblob import TextBlob
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer
import re

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import urllib.request
import zipfile
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Bidirectional
from tensorflow.keras.callbacks import History

# Step 1: Load the dataset
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df

# Step 1.1: Convert positive labels to 1 and negative labels to 0
def convert_labels(df):
    df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    return df

# Step 1.2: Remove duplicate rows
def remove_duplicates(df):
    df.drop_duplicates(inplace=True)

# Step 1.3: Randomly select 30% of the dataset
def random_selection(df):
    df = df.sample(frac=0.3, random_state=42)
    return df

# Step 1.4: Calculate polarity using TextBlob
def get_polarity(df):
    print('step-1-4')
    df['polarity'] = df['review'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    return df

# Step 1.5: Separate positive and negative polarities, plot histograms, and analyze means
def plot_sentiment_histograms(df):
    print('step-1-5')
    positive_reviews = df[df['sentiment'] == 1]
    negative_reviews = df[df['sentiment'] == 0]

    # Plotting histogram for positive sentiment
    plt.hist(positive_reviews['polarity'], bins=20, color='green', alpha=0.7)
    plt.title('positive')
    plt.xlabel('sense')
    plt.ylabel('count')
    plt.savefig('positive.png')
    plt.show()

    # Plotting histogram for negative sentiment
    plt.hist(negative_reviews['polarity'], bins=20, color='red', alpha=0.7)
    plt.title('negative')
    plt.xlabel('sense')
    plt.ylabel('count')
    plt.savefig('negative.png')
    plt.show()

    # Calculate mean polarity for positive and negative reviews
    mean_polarity_positive = positive_reviews['polarity'].mean()
    mean_polarity_negative = negative_reviews['polarity'].mean()

    # Print mean polarities
    print("Mean Polarity for Positive Reviews:", mean_polarity_positive)
    print("Mean Polarity for Negative Reviews:", mean_polarity_negative)

# Step 1.6: Apply preprocessing steps
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])

    stemmer = PorterStemmer()
    text = ' '.join([stemmer.stem(word) for word in text.split()])

# Step 1.7: Split the dataset into training and testing sets
def split_dataset(df):
    train_data, test_data, train_labels, test_labels = train_test_split(
        df['review'], df['sentiment'], test_size=0.5, random_state=42
    )
    return train_data, test_data, train_labels, test_labels


# Step 2-1: Create Bag of Words (BOW) representation for each review
def create_bow_representation(reviews_train, reviews_test):
    vectorizer = CountVectorizer()
    
    bow_matrix_train = vectorizer.fit_transform(reviews_train)
    bow_matrix_test = vectorizer.transform(reviews_test)
    
    return bow_matrix_train, bow_matrix_test

# Step 2-2: Cluster the training and testing sets using K-means
def kmeans_clustering(bow_matrix_train, bow_matrix_test, num_clusters=3):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    
    # Fit on training set
    train_clusters = kmeans.fit_predict(bow_matrix_train)

    # Predict on testing set
    test_clusters = kmeans.predict(bow_matrix_test)

    return train_clusters, test_clusters

# Step 2-3: Visualize clusters using t-sne
def visualize_clusters(bow_matrix, clusters, title):
    print('step-2-3')
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(bow_matrix.toarray())

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=clusters, cmap='viridis')
    plt.title(title)
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.savefig('training_clusters_plot.png')
    plt.show()

# Step 2-4: Select a uniform sample from each cluster
def select_uniform_sample(bow_matrix, clusters, num_samples):
    unique_clusters, cluster_counts = np.unique(clusters, return_counts=True)
    
    selected_samples = []

    for cluster in unique_clusters:
        cluster_indices = np.where(clusters == cluster)[0]
        mi = min(num_samples, cluster_counts[cluster])
        selected_indices = np.random.choice(cluster_indices, size=mi, replace=False)
        selected_samples.extend(selected_indices)

    return selected_samples


# Step 3-1: Convert text sequences to numerical sequences using Tokenizer
def tokenize_text_sequences(X_train, X_test):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)

    # Convert text sequences to numerical sequences
    train_sequences = tokenizer.texts_to_sequences(X_train)
    test_sequences = tokenizer.texts_to_sequences(X_test)

    return tokenizer, train_sequences, test_sequences

# Step 3-2: Pad sequences to have a consistent length
def pad_text_sequences(train_sequences, test_sequences):
    max_length = max(max(len(seq) for seq in train_sequences), max(len(seq) for seq in test_sequences))
    
    # Pad sequences
    padded_train_sequences = pad_sequences(train_sequences, maxlen=max_length, padding='post')
    padded_test_sequences = pad_sequences(test_sequences, maxlen=max_length, padding='post')

    return padded_train_sequences, padded_test_sequences, max_length


# Step 4: Download word embeddings and create an embedding matrix
def download_and_create_embedding_matrix(tokenizer):

    embedding_file_path = f'embeddings/wiki-news-300d-1M.vec'

    if not os.path.isfile(embedding_file_path):
        print('Downloading word vectors ... ')
        url = f'https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip'
        urllib.request.urlretrieve(url, f'wiki-news-300d-1M.vec.zip')

        print('Unzipping ... ')
        with zipfile.ZipFile(f'wiki-news-300d-1M.vec.zip', 'r') as zip_ref:
            zip_ref.extractall('embeddings')
        print('done.')

        os.remove(f'wiki-news-300d-1M.vec.zip')

    embeddings_index = {}
    with open(embedding_file_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, 300))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


# Step 5-1: Build and train a GRU model
def build_and_train_gru_model(embedding_matrix, max_length, input_dim):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=embedding_matrix.shape[1],
                        weights=[embedding_matrix], input_length=max_length, trainable=False))
    model.add(GRU(units=32, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(padded_train_sequences, y_train.iloc[selected_samples_train],
                        epochs=10, batch_size=64, validation_split=0.2)

    return model, history

# Step 5-2: Plot loss and accuracy for the GRU model
def plot_metrics(history, title):
    plt.figure(figsize=(12, 4))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(title + ' - Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(title + ' - Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()

# Step 5-3: Build and train a BiGRU model
def build_and_train_bigru_model(embedding_matrix, max_length, input_dim):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=embedding_matrix.shape[1],
                        weights=[embedding_matrix], input_length=max_length, trainable=False))
    model.add(Bidirectional(GRU(units=32, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(padded_train_sequences, y_train.iloc[selected_samples_train],
                        epochs=10, batch_size=64, validation_split=0.2)

    return model, history


file_path = 'IMDB_Dataset.csv'
df = load_dataset(file_path)

# step1-1
convert_labels(df)

# step1-2
remove_duplicates(df)

# step1-3
df = random_selection(df)

# step1-4
df = get_polarity(df)

# step1-5
plot_sentiment_histograms(df)

# step1-6
i = 0
for index, row in df.iterrows():
    print(i)
    i += 1
    df.at[index, 'review'] = preprocess_text(row['review'])

df.to_csv('preprocessed_and_split_IMDB_Dataset.csv', index=False)

# step1-7

file_path = 'preprocessed_and_split_IMDB_Dataset.csv'
df = load_dataset(file_path)
X_train, X_test, y_train, y_test = split_dataset(df)

# # step2-1
bow_matrix_train, bow_matrix_test = create_bow_representation(X_train, X_test)

# # Step 2-2
train_clusters, test_clusters = kmeans_clustering(bow_matrix_train, bow_matrix_test)

# # Step 2-3
visualize_clusters(bow_matrix_train, train_clusters, title='Training Set Clusters (t-SNE Visualization)')

# # Step 2-4
num_samples_per_cluster = 100  # Adjust the number of samples as needed
selected_samples_train = select_uniform_sample(bow_matrix_train, train_clusters, num_samples_per_cluster)
selected_samples_test = select_uniform_sample(bow_matrix_test, test_clusters, num_samples_per_cluster)

# # Step 3-1
tokenizer, train_sequences, test_sequences = tokenize_text_sequences(X_train.iloc[selected_samples_train], X_test.iloc[selected_samples_test])

# # Step 3-2
padded_train_sequences, padded_test_sequences, max_length = pad_text_sequences(train_sequences, test_sequences)

# step 4
embedding_matrix = download_and_create_embedding_matrix(tokenizer)

# Step 5-1 (GRU Model)
gru_model, gru_history = build_and_train_gru_model(embedding_matrix, max_length, input_dim=len(tokenizer.word_index) + 1)

# Step 5-2 (Plot Metrics for GRU Model)
plot_metrics(gru_history, 'GRU Model')

# Step 5-3 (BiGRU Model)
bigru_model, bigru_history = build_and_train_bigru_model(embedding_matrix, max_length, input_dim=len(tokenizer.word_index) + 1)

# Step 5-4 (Plot Metrics for BiGRU Model)
plot_metrics(bigru_history, 'BiGRU Model')