import pandas as pd
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# Pembacaan data ulasan
path = "Ulasan_Bebek_Ngarasan.xlsx"
dataUlasan = pd.read_excel(path)

# Preprocessing teks-teks ulasan
model = spacy.load("en_core_web_sm")

def textPreprocessing(text):
    if pd.isna(text):
        return ""
    processedText = model(text.lower()) 
    words = [token.lemma_ for token in processedText if not token.is_stop and not token.is_punct and len(token.text) > 3]  
    return " ".join(words)

processedDataUlasan = dataUlasan['textTranslated'].dropna().apply(textPreprocessing)

# Pembentukan matriks TF-IDF
vectorizer = TfidfVectorizer(
    max_features=220, 
    ngram_range=(1, 2),  
    max_df=0.75,  
    min_df=0.025 
)
matrixTFIDF = vectorizer.fit_transform(processedDataUlasan).toarray()

# LSA menggunakan SVD
U, S, Vt = np.linalg.svd(matrixTFIDF, full_matrices=False)
jumlahTopic = 3  
reducedVt = Vt[:jumlahTopic, :] 

# Kata-kata dari setiap topik
kata = vectorizer.get_feature_names_out()

print("\nTopik dan Kata-kata Dominan:")
for i, topic in enumerate(reducedVt):
    topKata = [kata[idx] for idx in np.argsort(topic)[::-1][:7]]  # Top 6 kata per topik
    print(f"Topik {i + 1}: {', '.join(topKata)}")
