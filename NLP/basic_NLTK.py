# !pip install nltk

import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize
print(word_tokenize(en_text))