import numpy as np
import re
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx

class TextPreprocessor:
    def __init__(self):
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except:
            pass

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z\s\.\!\?]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def safe_sent_tokenize(self, text):
        try:
            return sent_tokenize(str(text))
        except:
            return [str(text)] if str(text) else [""]

class HybridSummarizer:
    def __init__(self):
        self.preprocessor = TextPreprocessor()

    def summarize(self, text, num_sentences=3):
        try:
            sentences = self.preprocessor.safe_sent_tokenize(text)
            if len(sentences) <= num_sentences:
                return " ".join(sentences)

            # TF-IDF + TextRank logic (same as before)
            preprocessed_sentences = [self.preprocessor.clean_text(sent) for sent in sentences]
            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)
            tfidf_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()

            # Similarity matrix
            n = len(sentences)
            similarity_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i != j:
                        words_i = set(preprocessed_sentences[i].split())
                        words_j = set(preprocessed_sentences[j].split())
                        if words_i and words_j:
                            overlap = len(words_i.intersection(words_j))
                            union = len(words_i.union(words_j))
                            similarity_matrix[i][j] = overlap / union if union > 0 else 0

            nx_graph = nx.from_numpy_array(similarity_matrix)
            textrank_scores_dict = nx.pagerank(nx_graph)
            textrank_scores = [textrank_scores_dict[i] for i in range(len(sentences))]

            # Combine scores
            if np.max(tfidf_scores) > 0:
                tfidf_scores = tfidf_scores / np.max(tfidf_scores)
            if np.max(textrank_scores) > 0:
                textrank_scores = textrank_scores / np.max(textrank_scores)

            combined_scores = 0.5 * tfidf_scores + 0.5 * np.array(textrank_scores)
            top_indices = combined_scores.argsort()[-num_sentences:][::-1]
            top_sentences = [sentences[i] for i in sorted(top_indices)]

            return " ".join(top_sentences)

        except Exception as e:
            return f"Extractive summary: {text[:100]}..."