#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install wordcloud')


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

import re
from wordcloud import WordCloud

import nltk
nltk.download(['stopwords', 'punkt', 'wordnet', 'omw-1.4', 'vader_lexicon'])


# In[3]:


simple_text='This isn\'t a real text, this is an example text...Notice this contains punctuation!!'


# In[4]:


tokenizer = nltk.tokenize.RegexpTokenizer('[a-zA-Z0-9\']+')
tokenized_document = tokenizer.tokenize(simple_text)
print(tokenized_document)












# In[5]:


stop_words = nltk.corpus.stopwords.words('english')
print(stop_words)


# In[6]:


#remove stopwords
cleaned_tokens = []
for word in tokenized_document:
    word = word.lower()
    if word not in stop_words:
        cleaned_tokens.append(word)
print(cleaned_tokens)        


# In[7]:


# we can also remove stopwords using list comprehension

cleaned_tokens = [word.lower() for word in tokenized_document if word.lower() not in stop_words]
print(cleaned_tokens)


# In[8]:


# explore lemmatization vs stemming

lemmatizer = nltk.stem.WordNetLemmatizer()
stemmer = nltk.stem.PorterStemmer()

words = ['cacti','sings','hopped','rocks','better','easily']
pos = ['n','v','v','n','a','r']
lemmatized_words = [lemmatizer.lemmatize(words[i], pos=pos[i]) for i in range(6)]
stemmed_words = [stemmer.stem(word) for word in words]

print(" Lemmatized words: ", lemmatized_words)
print("Stemmed words: ", stemmed_words)


# In[9]:


from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
# Create a stemming object
stemmer = PorterStemmer()
cleaned_tokens = [word.lower() for word in tokenized_document if word.lower() not in stop_words]
print(cleaned_tokens)
# Perform stemming on the tokens
stemmed_text = [stemmer.stem(word) for word in cleaned_tokens]
print(stemmed_text)



# In[10]:


#lets now create a function to apply all of our data preprocessing steps which we can then use on a corpus 

def preprocess_text(text):
    tokenized_document = nltk.tokenize.RegexpTokenizer('[a-zA-Z0-9\']+').tokenize(text)#tokenize
    cleaned_tokens = [word.lower() for word in tokenized_document if word.lower() not in stop_words]#remove
    stemmed_text = [nltk.stem.PorterStemmer().stem(word) for word in cleaned_tokens]#stemming
    return stemmed_text








# In[11]:


data = pd.read_csv("ToT.txt", sep="\t", header=None, skiprows=1)
data.columns = ["ID","QuestionBody","Sentence","targetMovie","targetContext","targetSearch","targetSocial","segmentedSentence","hedging","opinionEmotion","absoluteRelative","flaggedContent","notes","threadURL","wikipediaURL","imdbURL"]


# In[12]:


data.head()


# In[13]:


print(data.info())


# In[14]:


# additional eda analysis
#Visualize Document Lengths:
# Assuming 'data' is your DataFrame and 'doc_length' is the document length column
sns.histplot(data['targetMovie'], kde=True)
plt.title('Distribution of Document Length')
plt.xlabel('Document Length')
plt.ylabel('Frequency')
plt.show()


# In[15]:


#Explore N-grams:
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import matplotlib.pyplot as plt
# Assuming 'data' is your DataFrame and 'processed_text' is the preprocessed text column
corpus = data['Sentence'].tolist()
# Assuming 'corpus' is your list of text data
corpus = [str(text) for text in corpus]  # Convert NaN to empty string or another suitable value

# Assuming 'corpus' is your list of text data
vectorizer = CountVectorizer(ngram_range=(1, 2))  # Adjust n-gram range as needed
X = vectorizer.fit_transform(corpus)

# Calculate the sum of occurrences for each n-gram
ngrams = pd.DataFrame(X.sum(axis=0), columns=vectorizer.get_feature_names_out()).transpose().sort_values(0, ascending=False)

# Plot the top N n-grams
top_n = 20
ngrams.head(top_n).plot(kind='bar', legend=False)
plt.title(f'Top {top_n} N-grams in the Corpus')
plt.xlabel('N-grams')
plt.ylabel('Frequency')
plt.show()


# In[16]:


#Visualize Document Similarity:

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Assuming 'corpus' is your list of preprocessed text data
corpus = ["your", "preprocessed", "text", "data"]

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

# Calculate cosine similarity between documents
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Visualize as a heatmap
sns.heatmap(cosine_sim, cmap='Blues', annot=False)
plt.title('Cosine Similarity between Documents')
plt.show()


# In[43]:


# Drop non-numeric columns
numeric_data = data.select_dtypes(include='number')

# Correlation Matrix
correlation_matrix = numeric_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


# In[55]:


# Feature Importance

# Identify important words or features using TF-IDF
feature_names = tfidf_vectorizer.get_feature_names_out()
feature_importance = pd.DataFrame(tfidf_matrix.sum(axis=0), columns=feature_names).transpose().sort_values(0, ascending=False)

# Display top N important words
top_n_words = 10
print(f"Top {top_n_words} Important Words:")
print(feature_importance.head(top_n_words))

# You can also visualize the top words
feature_importance.head(top_n_words).plot(kind='bar', legend=False)
plt.title(f'Top {top_n_words} Important Words')
plt.xlabel('Words')
plt.ylabel('TF-IDF Score')
plt.show()



# In[ ]:





# In[ ]:





# In[44]:


print(data.groupby("absoluteRelative").count())


# In[45]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if isinstance(text, str):  # Check if the text is a string
        words = word_tokenize(text)
        words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
        return ' '.join(words)
    else:
        return ''  # Return an empty string for NaN values


# In[46]:


#bag of words model
data['Sentence']= data['Sentence'].apply(preprocess_text)
data.head()


# In[47]:


#fixed-length vector
from sklearn.feature_extraction.text import CountVectorizer

# Preprocess the text and join the words
data['processed_text'] = data['Sentence'].apply(lambda x: ' '.join(x.lower() for x in x.split()))


vectorizer = CountVectorizer()
x=vectorizer.fit_transform(data['processed_text'])
x=pd.DataFrame(x.toarray())
x.head()





# In[56]:


from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import seaborn as sns
# Assuming 'x' is your feature data, and 'data' is your DataFrame
x = data.drop('absoluteRelative', axis=1)  # Assuming 'absoluteRelative' is your target column
y = data['absoluteRelative']
# Handle NaN values in the target variable
y = y.fillna('unknown')  # Replace NaN values with a placeholder or handle differently
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, test_size=0.2, random_state=99
)

resampler = RandomUnderSampler(random_state=0)
x_train_undersampled, y_train_undersampled = resampler.fit_resample(x_train, y_train)
sns.countplot(x=y_train_undersampled)


# In[49]:


from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import seaborn as sns

# Assuming 'x' is your feature data, and 'data' is your DataFrame
x = data['processed_text']  # Assuming 'processed_text' is the preprocessed text column
y = data['absoluteRelative']

# Handle NaN values in the target variable
y = y.fillna('unknown')  # Replace NaN values with a placeholder or handle differently

# Use LabelEncoder to convert string labels to numeric labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y_encoded, train_size=0.8, test_size=0.2, random_state=99
)

# Resample the training data to handle class imbalance
resampler = RandomUnderSampler(random_state=0)
x_train_undersampled, y_train_undersampled = resampler.fit_resample(x_train.values.reshape(-1, 1), y_train)

# Convert the text data to a bag-of-words representation using CountVectorizer
vectorizer = CountVectorizer()
x_train_bow = vectorizer.fit_transform(x_train_undersampled.flatten())
x_test_bow = vectorizer.transform(x_test)

# Train a simple classifier (Naive Bayes as an example)
model = MultinomialNB()
model.fit(x_train_bow, y_train_undersampled)

# Make predictions on the test set
predictions = model.predict(x_test_bow)

# Evaluate the model
accuracy = model.score(x_test_bow, y_test)
print(f"Accuracy: {accuracy}")


# In[50]:


from sklearn.metrics import confusion_matrix, classification_report
# Evaluate the model
accuracy = model.score(x_test_bow, y_test)
print(f"Accuracy: {accuracy}")
# Make predictions on the test set
predictions = model.predict(x_test_bow)
# Confusion Matrix
conf_matrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(conf_matrix)
# Classification Report
class_report = classification_report(y_test, predictions)
print("Classification Report:")
print(class_report)


# Sentiment Analysis
# 

# In[24]:


print(preprocess_text('This movie is great!'))
print(preprocess_text('This movie is not great'))


# In[25]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer

sentiment = SentimentIntensityAnalyzer()

print(sentiment.polarity_scores('This move is great!'))
print(sentiment.polarity_scores('This move is not great'))


# In[26]:


# Assuming 'data' is your DataFrame and 'Sentence' is the text column
data['processed_text'] = data['Sentence'].apply(preprocess_text)





# In[27]:


from textblob import TextBlob

# Assuming 'data' is your DataFrame and 'processed_text' is the preprocessed text column
data['sentiment'] = data['processed_text'].apply(lambda x: TextBlob(x).sentiment.polarity)










# In[28]:


pip install vaderSentiment


# In[29]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize the analyzer
analyzer = SentimentIntensityAnalyzer()

# Assuming 'data' is your DataFrame and 'processed_text' is the preprocessed text column
data['compound'] = data['processed_text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

# Categorize sentiments based on compound scores
data['sentiment_vader'] = data['compound'].apply(lambda x: 'positive' if x >= 0 else 'negative')







# In[30]:


import matplotlib.pyplot as plt

# Assuming 'data' is your DataFrame
plt.figure(figsize=(8, 6))
data['sentiment'].hist(bins=20, color='blue', alpha=0.7)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment Polarity')
plt.ylabel('Frequency')
plt.show()


# In[31]:


data.describe()


# In[32]:


pip install nltk


# In[33]:


import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download the VADER lexicon (if not already downloaded)
nltk.download('vader_lexicon')

# Initialize the analyzer
nltk_analyzer = SentimentIntensityAnalyzer()

# Assuming 'data' is your DataFrame and 'processed_text' is the preprocessed text column
data['nltk_compound'] = data['processed_text'].apply(lambda x: nltk_analyzer.polarity_scores(x)['compound'])
data['sentiment_nltk'] = data['nltk_compound'].apply(lambda x: 'positive' if x >= 0 else 'negative')

# Optionally, you can categorize sentiments into more categories
# For example, you can consider scores between -0.2 and 0.2 as neutral
data['sentiment_nltk'] = pd.cut(data['nltk_compound'], bins=[-float('inf'), -0.2, 0.2, float('inf')], labels=['negative', 'neutral', 'positive'])


# In[34]:


data.describe()


# In[ ]:





# In[57]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[58]:


# Assuming 'data' is your DataFrame
sns.histplot(data['nltk_compound'], bins=20, kde=True)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment Score (Compound)')
plt.ylabel('Frequency')
plt.show()


# In[36]:


# Assuming 'data' is your DataFrame
sns.boxplot(x='absoluteRelative', y='nltk_compound', data=data)
plt.title('Sentiment Distribution across Absolute Relative Categories')
plt.xlabel('Absolute Relative Category')
plt.ylabel('Sentiment Score (Compound)')
plt.show()


# In[37]:


from wordcloud import WordCloud

# Assuming 'data' is your DataFrame
positive_text = ' '.join(data[data['sentiment_nltk'] == 'positive']['processed_text'])
negative_text = ' '.join(data[data['sentiment_nltk'] == 'negative']['processed_text'])

# Generate word clouds
positive_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
negative_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(negative_text)

# Plot the word clouds
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(positive_wordcloud, interpolation='bilinear')
plt.title('Positive Sentiment Word Cloud')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(negative_wordcloud, interpolation='bilinear')
plt.title('Negative Sentiment Word Cloud')
plt.axis('off')

plt.show()


# In[38]:


# Assuming 'data' is your DataFrame
sns.boxplot(x='targetMovie', y='compound', data=data)
plt.title('Sentiment Distribution across Movies')
plt.xlabel('Movie')
plt.ylabel('Sentiment Score (Compound)')
plt.show()



# In[39]:


from wordcloud import WordCloud

# Assuming 'data' is your DataFrame
all_text = ' '.join(data['processed_text'])

# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

# Plot the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Entire Dataset')
plt.axis('off')
plt.show()


# In[40]:


# Assuming 'data' is your DataFrame
sns.lineplot(x='targetContext', y='compound', data=data)
plt.title('Sentiment Trends across Contexts')
plt.xlabel('Context')
plt.ylabel('Sentiment Score (Compound)')
plt.show()


# In[41]:


# Assuming 'data' is your DataFrame
sns.scatterplot(x='targetSearch', y='compound', data=data)
plt.title('Sentiment Correlation with Search')
plt.xlabel('Search')
plt.ylabel('Sentiment Score (Compound)')
plt.show() n


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




