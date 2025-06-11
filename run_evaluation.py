from utils import load_corpus, load_summary_corpus, load_queries_and_answers
from retriever import BM25Retriever, SummaryBM25Retriever
from llm_wrapper import LLMWrapper
from evaluation import run_comprehensive_evaluation, print_evaluation_summary
import time


def main():
    # Configuration
    CORPUS_FILEPATH = "data/multihoprag_corpus.txt"
    SUMMARY_CORPUS_FILEPATH = "data_summary/multihoprag_corpus_summary.csv"
    QA_FILEPATH = "data/MultiHopRAG.json"
    NUM_SAMPLES = 50
    print("="*80)
    print("IR-CoT COMPREHENSIVE EVALUATION: ORIGINAL vs SUMMARY CORPUS")
    print("="*80)

    # Load datasets
    print("\n1. Loading datasets...")
    print("-" * 40)

    original_documents = load_corpus(CORPUS_FILEPATH)
    summary_documents = load_summary_corpus(SUMMARY_CORPUS_FILEPATH)
    qa_dataset = load_queries_and_answers(QA_FILEPATH)

    if not original_documents or not summary_documents or not qa_dataset:
        print("Error: Failed to load one or more datasets. Please check file paths.")
        return

    print(f"✓ Original corpus: {len(original_documents)} documents")
    print(f"✓ Summary corpus: {len(summary_documents)} documents")
    print(f"✓ QA dataset: {len(qa_dataset)} questions")

    # Initialize retrievers
    print("\n2. Initializing retrievers...")
    print("-" * 40)

    original_retriever = BM25Retriever(original_documents)
    summary_retriever = SummaryBM25Retriever(summary_documents)

    print("✓ Original BM25 retriever initialized")
    print("✓ Summary BM25 retriever initialized")

    # Initialize LLM
    print("\n3. Initializing LLM...")
    print("-" * 40)

    # You can modify this based on your preferred LLM setup
    try:
        llm = LLMWrapper(
            model_identifier="qwen3:8b",
            llm_type="ollama"
        )
        print("✓ LLM initialized successfully")
    except Exception as e:
        print(f"Warning: Could not initialize OpenAI LLM ({e})")
        print("Trying Ollama as fallback...")
        try:
            llm = LLMWrapper(
                model_identifier="llama2",  # or your preferred local model
                llm_type="ollama"
            )
            print("✓ Ollama LLM initialized successfully")
        except Exception as e2:
            print(f"Error: Could not initialize any LLM ({e2})")
            print(
                "Please ensure you have either OpenAI API key or Ollama running locally.")
            return

    # Run evaluations
    print(f"\n4. Running evaluations on {NUM_SAMPLES} samples...")
    print("-" * 40)

    # Evaluate Original Corpus
    print("\nEvaluating ORIGINAL CORPUS...")
    start_time = time.time()

    original_results, original_summary = run_comprehensive_evaluation(
        qa_dataset=qa_dataset,
        retriever=original_retriever,
        llm_wrapper=llm,
        num_samples=NUM_SAMPLES,
        similarity_threshold=0.7,  # Slightly lower threshold for better matching
        verbose=False  # Set to True for detailed progress
    )

    original_time = time.time() - start_time
    print(
        f"✓ Original corpus evaluation completed in {original_time:.2f} seconds")

    # Evaluate Summary Corpus
    print("\nEvaluating SUMMARY CORPUS...")
    start_time = time.time()

    summary_results, summary_summary = run_comprehensive_evaluation(
        qa_dataset=qa_dataset,
        retriever=summary_retriever,
        llm_wrapper=llm,
        num_samples=NUM_SAMPLES,
        similarity_threshold=0.7,  # Slightly lower threshold for better matching
        verbose=False  # Set to True for detailed progress
    )

    summary_time = time.time() - start_time
    print(
        f"✓ Summary corpus evaluation completed in {summary_time:.2f} seconds")

    # Print results
    print("\n5. EVALUATION RESULTS")
    print("=" * 80)

    print_evaluation_summary(original_summary, "ORIGINAL CORPUS")
    print_evaluation_summary(summary_summary, "SUMMARY CORPUS")

    # Comparison Analysis
    print("\n6. COMPARATIVE ANALYSIS")
    print("=" * 80)

    print("\nAnswer Quality Comparison:")
    print(f"  Original Corpus F1:  {original_summary['avg_answer_f1']:.4f}")
    print(f"  Summary Corpus F1:   {summary_summary['avg_answer_f1']:.4f}")
    print(
        f"  Difference:           {summary_summary['avg_answer_f1'] - original_summary['avg_answer_f1']:+.4f}")

    print(f"\nRetrieval Accuracy Comparison:")
    print(
        f"  Original Retrieval F1:  {original_summary['avg_retrieval_f1']:.4f}")
    print(
        f"  Summary Retrieval F1:   {summary_summary['avg_retrieval_f1']:.4f}")
    print(
        f"  Difference:             {summary_summary['avg_retrieval_f1'] - original_summary['avg_retrieval_f1']:+.4f}")

    print(f"\nEfficiency Comparison:")
    print(f"  Original Processing Time:  {original_time:.2f} seconds")
    print(f"  Summary Processing Time:   {summary_time:.2f} seconds")
    print(
        f"  Speed Improvement:         {((original_time - summary_time) / original_time * 100):+.1f}%")

    print(f"\nRetrieval Statistics:")
    print(
        f"  Original Match Rate:  {(original_summary['total_exact_matches'] + original_summary['total_fuzzy_matches']) / original_summary['total_retrieved']:.4f}")
    print(
        f"  Summary Match Rate:   {(summary_summary['total_exact_matches'] + summary_summary['total_fuzzy_matches']) / summary_summary['total_retrieved']:.4f}")

    # Detailed Analysis Examples
    print("\n7. DETAILED EXAMPLES")
    print("=" * 80)

    print("\nExample comparisons (first 3 questions):")
    for i in range(min(3, len(original_results))):
        print(f"\n--- Question {i+1} ---")
        print(f"Q: {original_results[i]['question'][:100]}...")
        print(f"Ground Truth: {original_results[i]['ground_truth_answer']}")

        print(f"\nOriginal Corpus:")
        print(f"  Answer: {original_results[i]['predicted_answer']}")
        print(f"  Answer F1: {original_results[i]['f1_score']:.4f}")
        print(f"  Retrieval F1: {original_results[i]['f1']:.4f}")
        print(f"  Retrieved Titles: {original_results[i]['retrieved_titles']}")

        print(f"\nSummary Corpus:")
        print(f"  Answer: {summary_results[i]['predicted_answer']}")
        print(f"  Answer F1: {summary_results[i]['f1_score']:.4f}")
        print(f"  Retrieval F1: {summary_results[i]['f1']:.4f}")
        print(f"  Retrieved Titles: {summary_results[i]['retrieved_titles']}")

    print("\n" + "="*80)
    print("EVALUATION COMPLETED")
    print("="*80)


def quick_test():
    """
    Quick test function for development/debugging.
    """
    print("Running quick test with 5 samples...")

    # Load datasets
    summary_documents = load_summary_corpus(
        "data_summary/multihoprag_corpus_summary.csv")
    qa_dataset = load_queries_and_answers("data/MultiHopRAG.json")

    # Initialize retriever
    summary_retriever = SummaryBM25Retriever(summary_documents)

    # Initialize LLM (placeholder - replace with your setup)
    llm = LLMWrapper(
        model_identifier="qwen3:8b",
        llm_type="ollama"
    )

    # Run quick evaluation
    results, summary = run_comprehensive_evaluation(
        qa_dataset=qa_dataset,
        retriever=summary_retriever,
        llm_wrapper=llm,
        num_samples=5,
        verbose=True,
        run_mode="multi-hop-ircot"
    )

    print_evaluation_summary(summary, "QUICK TEST")


if __name__ == "__main__":
    quick_test()
