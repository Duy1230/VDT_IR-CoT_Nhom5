from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize


class BM25Retriever:
    def __init__(self, documents_list):
        """
        Initializes the BM25 retriever.
        Args:
            documents_list (list): A list of document dictionaries, 
                                   each with 'id', 'title', and 'passage'.
        """
        self.documents = documents_list
        self.corpus_texts = [doc['passage']
                             # Or combine title + passage
                             for doc in documents_list]

        # Tokenize the corpus (simple whitespace and lowercasing for now)
        # You can use nltk.word_tokenize for more robust tokenization
        try:
            self.tokenized_corpus = [word_tokenize(
                doc.lower()) for doc in self.corpus_texts]
        except NameError:  # Fallback if nltk not available or punkt not downloaded
            print("NLTK word_tokenize not available. Using simple split(). Consider installing NLTK and downloading 'punkt'.")
            self.tokenized_corpus = [doc.lower().split()
                                     for doc in self.corpus_texts]

        if not self.tokenized_corpus or not any(self.tokenized_corpus):
            print("Warning: Tokenized corpus is empty or contains only empty documents. BM25 may not work correctly.")
            self.bm25 = None
        else:
            self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query_text, k=5):
        """
        Retrieves the top-k most relevant documents for a given query.
        Args:
            query_text (str): The query string.
            k (int): The number of top documents to retrieve.
        Returns:
            list: A list of tuples, where each tuple is (document_id, score, document_text, document_title).
                  Returns empty list if BM25 is not initialized.
        """
        if self.bm25 is None:
            print("Error: BM25 model not initialized, possibly due to empty corpus.")
            return []

        try:
            tokenized_query = word_tokenize(query_text.lower())
        except NameError:
            tokenized_query = query_text.lower().split()

        doc_scores = self.bm25.get_scores(tokenized_query)

        # Get top-k scores and their indices
        top_k_indices = sorted(range(len(doc_scores)),
                               key=lambda i: doc_scores[i], reverse=True)[:k]

        results = []
        for i in top_k_indices:
            doc = self.documents[i]
            results.append({
                "id": doc['id'],
                "score": doc_scores[i],
                "text": doc['passage'],  # Return the full passage
                "title": doc['title']
            })
        return results


class SummaryBM25Retriever:
    """
    BM25 Retriever specifically optimized for summarized corpus data.
    Uses both title and summarized passage for retrieval to improve relevance.
    """

    def __init__(self, documents_list):
        """
        Initializes the Summary BM25 retriever.
        Args:
            documents_list (list): A list of document dictionaries from load_summary_corpus(),
                                   each with 'id', 'title', and 'passage' (summarized).
        """
        self.documents = documents_list

        # Combine title and passage for better retrieval
        # Weight title more heavily by including it twice
        self.corpus_texts = [f"{doc['title']} {doc['title']} {doc['passage']}"
                             for doc in documents_list]

        # Tokenize the corpus
        try:
            self.tokenized_corpus = [word_tokenize(
                doc.lower()) for doc in self.corpus_texts]
        except NameError:  # Fallback if nltk not available or punkt not downloaded
            print("NLTK word_tokenize not available. Using simple split(). Consider installing NLTK and downloading 'punkt'.")
            self.tokenized_corpus = [doc.lower().split()
                                     for doc in self.corpus_texts]

        if not self.tokenized_corpus or not any(self.tokenized_corpus):
            print("Warning: Tokenized corpus is empty or contains only empty documents. BM25 may not work correctly.")
            self.bm25 = None
        else:
            self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query_text, k=5):
        """
        Retrieves the top-k most relevant documents for a given query.
        Args:
            query_text (str): The query string.
            k (int): The number of top documents to retrieve.
        Returns:
            list: A list of dictionaries with document information and scores.
                  Returns empty list if BM25 is not initialized.
        """
        if self.bm25 is None:
            print("Error: BM25 model not initialized, possibly due to empty corpus.")
            return []

        try:
            tokenized_query = word_tokenize(query_text.lower())
        except NameError:
            tokenized_query = query_text.lower().split()

        doc_scores = self.bm25.get_scores(tokenized_query)

        # Get top-k scores and their indices
        top_k_indices = sorted(range(len(doc_scores)),
                               key=lambda i: doc_scores[i], reverse=True)[:k]

        results = []
        for i in top_k_indices:
            doc = self.documents[i]
            results.append({
                "id": doc['id'],
                "score": doc_scores[i],
                "text": doc['passage'],  # Return the summarized passage
                "title": doc['title']
            })
        return results
