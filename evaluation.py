import re
from typing import List, Dict, Any, Tuple
from difflib import SequenceMatcher


def normalize_title(title: str) -> str:
    """
    Normalize title text for comparison by removing punctuation and converting to lowercase.
    """
    # Remove special characters and normalize whitespace
    normalized = re.sub(r'[^\w\s]', ' ', title.lower())
    normalized = ' '.join(normalized.split())
    return normalized


def calculate_title_similarity(title1: str, title2: str, threshold: float = 0.8) -> float:
    """
    Calculate similarity between two titles using sequence matching.
    Returns similarity score between 0 and 1.
    """
    norm_title1 = normalize_title(title1)
    norm_title2 = normalize_title(title2)

    # Use SequenceMatcher for fuzzy matching
    similarity = SequenceMatcher(None, norm_title1, norm_title2).ratio()
    return similarity


def extract_retrieved_titles(retrieved_context: str) -> List[str]:
    """
    Extract document titles from retrieved context string.
    """
    titles = []

    # Handle None or empty context
    if not retrieved_context:
        return titles

    # Split by title markers and extract titles
    lines = retrieved_context.split('\n')
    for line in lines:
        if line.strip().startswith('Title:'):
            title = line.replace('Title:', '').strip()
            if title:
                titles.append(title)

    return titles


def evaluate_retrieval_accuracy(
    retrieved_titles: List[str],
    ground_truth_evidence: List[Dict[str, Any]],
    similarity_threshold: float = 0.8
) -> Dict[str, Any]:
    """
    Evaluate retrieval accuracy by comparing retrieved titles to ground truth evidence.

    Args:
        retrieved_titles: List of retrieved document titles
        ground_truth_evidence: List of evidence dictionaries with 'title' field
        similarity_threshold: Minimum similarity score to consider a match

    Returns:
        Dictionary with evaluation metrics
    """
    if not retrieved_titles or not ground_truth_evidence:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "exact_matches": 0,
            "fuzzy_matches": 0,
            "total_retrieved": len(retrieved_titles),
            "total_ground_truth": len(ground_truth_evidence),
            "matched_titles": []
        }

    ground_truth_titles = [evidence['title']
                           for evidence in ground_truth_evidence]

    exact_matches = 0
    fuzzy_matches = 0
    matched_titles = []

    # Track which ground truth titles have been matched
    gt_matched = [False] * len(ground_truth_titles)

    for retrieved_title in retrieved_titles:
        best_match_idx = -1
        best_similarity = 0.0

        # Find best matching ground truth title
        for i, gt_title in enumerate(ground_truth_titles):
            if gt_matched[i]:  # Skip already matched titles
                continue

            similarity = calculate_title_similarity(retrieved_title, gt_title)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match_idx = i

        # Check if we found a good match
        if best_match_idx >= 0 and best_similarity >= similarity_threshold:
            gt_matched[best_match_idx] = True
            matched_titles.append({
                "retrieved": retrieved_title,
                "ground_truth": ground_truth_titles[best_match_idx],
                "similarity": best_similarity
            })

            if best_similarity >= 0.95:  # Very high similarity = exact match
                exact_matches += 1
            else:
                fuzzy_matches += 1

    # Calculate metrics
    total_matches = exact_matches + fuzzy_matches
    precision = total_matches / \
        len(retrieved_titles) if retrieved_titles else 0.0
    recall = total_matches / \
        len(ground_truth_titles) if ground_truth_titles else 0.0
    f1 = (2 * precision * recall) / (precision +
                                     recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_matches": exact_matches,
        "fuzzy_matches": fuzzy_matches,
        "total_retrieved": len(retrieved_titles),
        "total_ground_truth": len(ground_truth_titles),
        "matched_titles": matched_titles
    }


def evaluate_single_question(
    result: Dict[str, Any],
    ground_truth_evidence: List[Dict[str, Any]],
    similarity_threshold: float = 0.8
) -> Dict[str, Any]:
    """
    Evaluate a single question's retrieval performance.

    Args:
        result: Result dictionary from run_qa_system
        ground_truth_evidence: Ground truth evidence list from MultiHopRAG.json
        similarity_threshold: Minimum similarity for fuzzy matching

    Returns:
        Evaluation metrics dictionary
    """
    # Handle cases where retrieved_context might be None (e.g., llm-only mode)
    retrieved_context = result.get("retrieved_context", "") or ""
    retrieved_titles = extract_retrieved_titles(retrieved_context)

    retrieval_eval = evaluate_retrieval_accuracy(
        retrieved_titles,
        ground_truth_evidence,
        similarity_threshold
    )

    # Add question-level information
    retrieval_eval.update({
        "question": result.get("question", ""),
        "predicted_answer": result.get("predicted_answer", ""),
        "ground_truth_answer": result.get("ground_truth", ""),
        "f1_score": result.get("f1_score", 0.0),
        "hops_taken": result.get("hops_taken", 0),
        "retrieved_titles": retrieved_titles
    })

    return retrieval_eval


def run_comprehensive_evaluation(
    qa_dataset: List[Dict[str, Any]],
    retriever,
    llm_wrapper,
    num_samples: int = 200,
    similarity_threshold: float = 0.8,
    max_ircot_hops: int = 3,
    k_retrieve_single_hop: int = 5,
    k_retrieve_multi_hop: int = 2,
    run_mode: str = "multi-hop-ircot",
    verbose_level: int = 0,
    verbose: bool = True
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Run comprehensive evaluation on the dataset.

    Args:
        qa_dataset: List of questions with evidence
        retriever: Initialized retriever (BM25Retriever or SummaryBM25Retriever)
        llm_wrapper: Initialized LLM wrapper
        num_samples: Number of samples to evaluate
        similarity_threshold: Minimum similarity for fuzzy matching
        verbose: Whether to print progress


    Returns:
        Tuple of (detailed_results, summary_metrics)
    """
    from run_pipelines import run_qa_system

    results = []
    retrieval_metrics = []

    for i in range(min(num_samples, len(qa_dataset))):
        print(f"--- Evaluating Sample {i+1}/{num_samples} ---")

        sample_data = qa_dataset[i]

        # Run IR-CoT
        result = run_qa_system(
            sample_data,
            retriever,
            llm_wrapper,
            mode=run_mode,
            max_ircot_hops=max_ircot_hops,
            k_retrieve_single_hop=k_retrieve_single_hop,
            k_retrieve_multi_hop=k_retrieve_multi_hop,
            verbose_level=verbose_level  # Silent
        )

        # Evaluate retrieval
        ground_truth_evidence = sample_data.get("evidence_list", [])
        eval_result = evaluate_single_question(
            result,
            ground_truth_evidence,
            similarity_threshold
        )

        results.append(eval_result)
        retrieval_metrics.append(eval_result)

        if verbose:
            print(f"-Answer")
            print(f"    Question: {eval_result['question'][:100]}...")
            print(f"    Predicted: {eval_result['predicted_answer']}")
            print(f"    Ground Truth: {eval_result['ground_truth_answer']}")
            print(f"    Answer F1: {eval_result['f1_score']:.4f}")
            print(f"    Hops Taken: {eval_result['hops_taken']}")

            # Only show retrieval metrics if there was actually retrieval
            if eval_result['total_retrieved'] > 0 or run_mode != "llm-only":
                print(f"-Retrieval")
                print(f"    Precision: {eval_result['precision']:.4f}")
                print(f"    Recall: {eval_result['recall']:.4f}")
                print(f"    F1: {eval_result['f1']:.4f}")
                print(f"    Retrieved {eval_result['total_retrieved']} docs, "
                      f"matched {eval_result['exact_matches'] + eval_result['fuzzy_matches']} "
                      f"out of {eval_result['total_ground_truth']} ground truth")
            else:
                print(f"-Retrieval: No retrieval performed (LLM-only mode)")
            print()

    # Calculate summary metrics
    summary = {
        "avg_answer_f1": sum(r['f1_score'] for r in results) / len(results) if results else 0.0,
        "avg_hops": sum(r['hops_taken'] for r in results) / len(results) if results else 0.0,
        "avg_retrieval_precision": sum(r['precision'] for r in retrieval_metrics) / len(retrieval_metrics) if retrieval_metrics else 0.0,
        "avg_retrieval_recall": sum(r['recall'] for r in retrieval_metrics) / len(retrieval_metrics) if retrieval_metrics else 0.0,
        "avg_retrieval_f1": sum(r['f1'] for r in retrieval_metrics) / len(retrieval_metrics) if retrieval_metrics else 0.0,
        "total_exact_matches": sum(r['exact_matches'] for r in retrieval_metrics),
        "total_fuzzy_matches": sum(r['fuzzy_matches'] for r in retrieval_metrics),
        "total_retrieved": sum(r['total_retrieved'] for r in retrieval_metrics),
        "total_ground_truth": sum(r['total_ground_truth'] for r in retrieval_metrics),
        "num_samples": len(results),
        "run_mode": run_mode
    }

    return results, summary


def print_evaluation_summary(summary: Dict[str, float], dataset_name: str = ""):
    """
    Print a formatted summary of evaluation results.
    """
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY - {dataset_name}")
    print(f"{'='*60}")
    print(f"Samples Evaluated: {summary['num_samples']}")
    print(f"Mode: {summary.get('run_mode', 'Unknown')}")
    print(f"\nAnswer Quality:")
    print(f"  Average Answer F1 Score: {summary['avg_answer_f1']:.4f}")
    print(f"  Average Number of Hops: {summary['avg_hops']:.2f}")

    # Only show retrieval metrics if there was retrieval
    if summary.get('run_mode') != 'llm-only' and summary['total_retrieved'] > 0:
        print(f"\nRetrieval Performance:")
        print(
            f"  Average Retrieval Precision: {summary['avg_retrieval_precision']:.4f}")
        print(
            f"  Average Retrieval Recall: {summary['avg_retrieval_recall']:.4f}")
        print(f"  Average Retrieval F1: {summary['avg_retrieval_f1']:.4f}")
        print(f"\nRetrieval Statistics:")
        print(f"  Total Documents Retrieved: {summary['total_retrieved']}")
        print(
            f"  Total Ground Truth Documents: {summary['total_ground_truth']}")
        print(f"  Total Exact Matches: {summary['total_exact_matches']}")
        print(f"  Total Fuzzy Matches: {summary['total_fuzzy_matches']}")
        if summary['total_retrieved'] > 0:
            print(
                f"  Overall Match Rate: {(summary['total_exact_matches'] + summary['total_fuzzy_matches']) / summary['total_retrieved']:.4f}")
        else:
            print(f"  Overall Match Rate: N/A (no documents retrieved)")
    else:
        print(f"\nRetrieval Performance: N/A (LLM-only mode - no retrieval performed)")

    print(f"{'='*60}")
