import nltk
#nltk.download('punkt')
#nltk.download('wordnet')
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

#create an object of class PorterStemmer
porter = PorterStemmer()
#proide a word to be stemmed
print("Porter Stemmer")
print(porter.stem("cats"))
print(porter.stem("trouble"))
print(porter.stem("troubling"))
print(porter.stem("troubled"))

print(porter.stem("connect"))
print(porter.stem("connected"))
print(porter.stem("connection"))

wordnet_lemmatizer = WordNetLemmatizer()
print(wordnet_lemmatizer.lemmatize("connect"))
print(wordnet_lemmatizer.lemmatize("connected"))
print(wordnet_lemmatizer.lemmatize("connection"))
print(wordnet_lemmatizer.lemmatize("connections"))