import json
import re
from nltk.tokenize import word_tokenize  # Using nltk for tokenization for BM25


def load_corpus(corpus_filepath: str = "multihoprag_corpus.txt") -> list[dict]:
    """
    Loads and parses the corpus file.
    Each document is expected to have a 'Title:' and 'Passage:'
    and be separated by '<endofpassage>'.

    Returns:
        list: A list of dictionaries, where each dict has 'id', 'title', and 'passage'.
    """
    documents = []
    doc_id_counter = 0
    try:
        with open(corpus_filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split by the endofpassage marker, ensuring we handle potential extra newlines
        raw_docs = content.strip().split('<endofpassage>')

        for raw_doc in raw_docs:
            raw_doc = raw_doc.strip()  # Remove leading/trailing whitespace
            if not raw_doc:  # Skip empty entries that might result from splitting
                continue

            title_match = re.search(r"Title:\s*(.*)", raw_doc, re.IGNORECASE)
            # DOTALL to match newlines in passage
            passage_match = re.search(
                r"Passage:\s*(.*)", raw_doc, re.DOTALL | re.IGNORECASE)

            title = title_match.group(1).strip() if title_match else "N/A"
            passage_text = passage_match.group(
                1).strip() if passage_match else ""

            # Clean up passage text from the title part if it was captured by DOTALL
            if passage_match and title_match and passage_text.startswith(title):
                # A bit hacky, but handles cases where passage regex grabs too much
                # Find the actual start of the passage content after "Passage:"
                passage_start_index = raw_doc.lower().find("passage:")
                if passage_start_index != -1:
                    passage_text = raw_doc[passage_start_index +
                                           len("passage:"):].strip()

            if title and passage_text:  # Ensure both title and passage are present
                documents.append({
                    "id": doc_id_counter,
                    "title": title,
                    "passage": passage_text
                })
                doc_id_counter += 1
            elif title and not passage_text:
                # Handle cases where a title might exist but passage is empty or not found
                print(
                    f"Warning: Document with title '{title}' has no passage content or parsing error.")
                documents.append({
                    "id": doc_id_counter,
                    "title": title,
                    "passage": ""  # Store with empty passage
                })
                doc_id_counter += 1
            # else:
                # print(f"Warning: Could not parse a document segment properly:\n---\n{raw_doc[:200]}...\n---")

    except FileNotFoundError:
        print(f"Error: Corpus file not found at {corpus_filepath}")
        return []

    print(f"Loaded {len(documents)} documents from corpus.")
    return documents


def load_queries_and_answers(qa_filepath="MultiHopRAG.json"):
    """
    Loads the JSON file containing queries, answers, and evidence.

    Returns:
        list: A list of dictionaries, each representing a query-answer item.
    """
    try:
        with open(qa_filepath, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
        print(f"Loaded {len(qa_data)} query-answer pairs.")
        return qa_data
    except FileNotFoundError:
        print(f"Error: QA file not found at {qa_filepath}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {qa_filepath}")
        return []


def load_summary_corpus(corpus_filepath: str = "data_summary/multihoprag_corpus_summary.csv") -> list[dict]:
    """
    Loads and parses the summarized corpus CSV file.
    Each row contains title, passage (summarized), and content (full formatted text).

    Returns:
        list: A list of dictionaries, where each dict has 'id', 'title', and 'passage'.
    """
    import csv

    documents = []
    doc_id_counter = 0

    try:
        with open(corpus_filepath, 'r', encoding='utf-8') as f:
            csv_reader = csv.DictReader(f)

            for row in csv_reader:
                title = row.get('title', '').strip()
                # Use the summarized passage
                passage = row.get('passage', '').strip()

                if title and passage:
                    documents.append({
                        "id": doc_id_counter,
                        "title": title,
                        "passage": passage
                    })
                    doc_id_counter += 1
                else:
                    print(
                        f"Warning: Skipping row with missing title or passage: {row}")

    except FileNotFoundError:
        print(f"Error: Summary corpus file not found at {corpus_filepath}")
        return []
    except Exception as e:
        print(f"Error loading summary corpus: {e}")
        return []

    print(f"Loaded {len(documents)} documents from summary corpus.")
    return documents
