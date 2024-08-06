import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer



nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def normalize_text(text):
    text = text.lower()
    word_tokens = word_tokenize(text)
    word_tokens = [word for word in word_tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    word_tokens = [word for word in word_tokens if not word in stop_words]
    lemmatizer = WordNetLemmatizer()
    word_tokens = [lemmatizer.lemmatize(word) for word in word_tokens]
    return " ".join(word_tokens)

# load your csv file
df = pd.read_csv('data_final_openend.csv')

# assume that 'your_column' is the column you want to apply your function on
# apply function to each row in 'your_column'
df['new_response'] = df['response'].apply(normalize_text)

# save the result in a new csv file
df.to_csv('normalized_file.csv', index=False)