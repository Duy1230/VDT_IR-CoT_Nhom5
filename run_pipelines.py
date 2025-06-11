from metric import calculate_f1_score
from llm_wrapper import LLMWrapper
from retriever import BM25Retriever, SummaryBM25Retriever
from prompts import construct_cot_step_prompt, construct_final_answer_prompt
from typing import Union

# Type alias for retriever that accepts both types
RetrieverType = Union[BM25Retriever, SummaryBM25Retriever]


def _format_retrieved_context(retrieved_docs, use_endofpassage=True):
    """
    Helper function to format retrieved documents into context string.
    Args:
        retrieved_docs: List of retrieved documents
        use_endofpassage: Whether to add <endofpassage> markers (for original corpus)
    """
    context_str = ""
    if retrieved_docs:
        for doc in retrieved_docs:
            if use_endofpassage:
                context_str += f"Title: {doc['title']}\nPassage: {doc['text']}\n<endofpassage>\n"
            else:
                context_str += f"Title: {doc['title']}\nPassage: {doc['text']}\n\n"
    return context_str


def answer_question_llm_only(
    original_question: str,
    llm_wrapper: LLMWrapper,
    max_final_answer_tokens: int = 100,
    verbose: bool = True
) -> tuple[str, int]:
    """
    Answers a question using only the LLM's parametric knowledge (no retrieval).
    Returns: (predicted_answer, 0 hops)
    """
    if verbose:
        print(f"\n{'='*15} Mode 0: LLM Only (No Retrieval) {'='*15}")
        print(f"Original Question: {original_question}")

    # Simple prompt for direct answering without context
    prompt = f"""
    Answer the following question based on your general knowledge.
    Question: {original_question}
    DIRECT ANSWER ONLY NO EXPLANATION('Yes','No', Name, Place, Number, etc.)
    Answer: """

    predicted_answer = llm_wrapper.generate(
        prompt,
        max_new_tokens=max_final_answer_tokens,
        temperature=0.3  # Factual answer
    )
    if verbose:
        print(f"LLM Only Predicted Answer: {predicted_answer}")
    return predicted_answer, 0  # 0 hops


def answer_question_single_hop_rag(
    original_question: str,
    retriever: RetrieverType,
    llm_wrapper: LLMWrapper,
    k_retrieve: int = 2,  # Number of documents for single retrieval
    max_final_answer_tokens: int = 100,
    verbose: bool = True
) -> tuple[str, int, str]:
    """
    Answers a question using single-hop RAG.
    Retrieves once, then generates answer.
    Returns: (predicted_answer, 1 hop, retrieved_context_str)
    """
    if verbose:
        print(f"\n{'='*15} Mode 1: Single-Hop RAG {'='*15}")
        print(f"Original Question: {original_question}")

    retrieved_docs = retriever.search(original_question, k=k_retrieve)

    # Determine if we're using original corpus (has <endofpassage>) or summary corpus
    use_endofpassage = isinstance(retriever, BM25Retriever)
    retrieved_context_str = _format_retrieved_context(
        retrieved_docs, use_endofpassage)

    if verbose:
        if retrieved_docs:
            print(f"Retrieved {len(retrieved_docs)} documents for single hop.")
        else:
            print("No documents retrieved for single hop.")

    prompt = f"""Based on the following retrieved context, answer the question.
                If the context is not sufficient, answer based on your general knowledge.
                Question: {original_question}
                Retrieved Context:
                ---
                {retrieved_context_str if retrieved_context_str else "No context was retrieved."}
                ---
                DIRECT ANSWER ONLY NO EXPLANATION('Yes','No', Name, Place, Number, etc.)
                Answer: """

    predicted_answer = llm_wrapper.generate(
        prompt,
        max_new_tokens=max_final_answer_tokens,
        temperature=0.3
    )
    if verbose:
        print(f"Single-Hop RAG Predicted Answer: {predicted_answer}")
    return predicted_answer, 1, retrieved_context_str  # 1 hop


def answer_question_multi_hop_ircot(
    original_question: str,
    retriever: RetrieverType,
    llm_wrapper: LLMWrapper,
    max_hops: int = 3,
    k_retrieve_per_hop: int = 2,
    max_cot_step_tokens: int = 100,
    max_final_answer_tokens: int = 100,
    confidence: bool = True,
    verbose: bool = True
) -> tuple[str, int, list[str], str]:
    """
    Returns: (final_answer_text, num_hops_taken, full_chain_of_thought_list, full_accumulated_context_str)
    """
    if verbose:
        print(f"\n{'='*15} Mode N: Multi-Hop IR-CoT (Max Hops: {max_hops}) {'='*15}")

    # Get context allocation for this model
    context_allocation = llm_wrapper.get_recommended_context_allocation()
    max_cot_context_tokens = context_allocation["cot_step_context"]
    max_final_context_tokens = context_allocation["final_answer_context"]
    # Rough char estimate
    max_accumulated_context_chars = context_allocation["max_accumulated_context"] * 3

    if verbose:
        print(f"Model: {llm_wrapper.model_identifier}")
        print(f"Context limit: {llm_wrapper.get_context_limit()} tokens")
        print(f"CoT context allocation: {max_cot_context_tokens} tokens")
        print(
            f"Final answer context allocation: {max_final_context_tokens} tokens")

    if verbose:
        print(f"Original Question: {original_question}\n")

    accumulated_context_str = ""
    full_chain_of_thought_list = []
    num_actual_hops_taken = 0  # This will count actual reasoning/retrieval cycles

    # Determine if we're using original corpus (has <endofpassage>) or summary corpus
    use_endofpassage = isinstance(retriever, BM25Retriever)

    if verbose:
        print(f"--- Initial Retrieval (considered pre-hop) ---")
    current_query_for_retrieval = original_question
    initial_retrieved_docs = retriever.search(
        current_query_for_retrieval, k=k_retrieve_per_hop)

    if initial_retrieved_docs:
        accumulated_context_str = _format_retrieved_context(
            initial_retrieved_docs, use_endofpassage)
        if verbose:
            print(
                f"Retrieved {len(initial_retrieved_docs)} initial documents for query: '{current_query_for_retrieval}'")
    else:
        if verbose:
            print(
                f"No initial documents retrieved for query: '{current_query_for_retrieval}'\n")

    for hop_iter_num in range(max_hops):  # Iterate up to max_hops
        num_actual_hops_taken = hop_iter_num + 1
        if verbose:
            print(
                f"\n--- IR-CoT Hop Iteration {num_actual_hops_taken}/{max_hops} ---")

        # Truncate accumulated context if it's getting too long
        if len(accumulated_context_str) > max_accumulated_context_chars:
            if verbose:
                print(
                    f"Truncating accumulated context from {len(accumulated_context_str)} to {max_accumulated_context_chars} chars")
            accumulated_context_str = accumulated_context_str[:
                                                              max_accumulated_context_chars] + "\n[Context Truncated]"

        cot_prompt = construct_cot_step_prompt(
            original_question=original_question,
            accumulated_context=accumulated_context_str,
            cot_so_far=full_chain_of_thought_list,
            max_context_tokens=max_cot_context_tokens  # Use dynamic context limit
        )

        if verbose:
            print(f"Generating CoT step {num_actual_hops_taken}...")
        next_cot_sentence = llm_wrapper.generate(
            cot_prompt, max_new_tokens=max_cot_step_tokens, temperature=0.5
        )
        if verbose:
            print(
                f"CoT Step {num_actual_hops_taken} Output: {next_cot_sentence}")
        full_chain_of_thought_list.append(next_cot_sentence)

        if next_cot_sentence.lower().startswith("the answer is:"):
            if verbose:
                print(
                    "LLM indicated answer in CoT step. Will proceed to final answer generation.")
            current_query_for_retrieval = None  # Flag not to retrieve
            if confidence:
                break
        else:
            current_query_for_retrieval = next_cot_sentence

        # Only retrieve if not the last hop and query exists
        if current_query_for_retrieval and (hop_iter_num < max_hops - 1):
            if verbose:
                print(
                    f"Preparing to retrieve based on: '{current_query_for_retrieval[:100]}...'")
            newly_retrieved_docs = retriever.search(
                current_query_for_retrieval, k=k_retrieve_per_hop)
            if newly_retrieved_docs:
                added_new_context = False
                for doc in newly_retrieved_docs:
                    if use_endofpassage:
                        doc_full_text = f"Title: {doc['title']}\nPassage: {doc['text']}\n<endofpassage>\n"
                    else:
                        doc_full_text = f"Title: {doc['title']}\nPassage: {doc['text']}\n\n"

                    if doc_full_text not in accumulated_context_str:
                        # Check if adding this would exceed our context limit
                        if len(accumulated_context_str) + len(doc_full_text) <= max_accumulated_context_chars:
                            accumulated_context_str += doc_full_text
                            added_new_context = True
                        elif verbose:
                            print(f"Skipping document due to context limit")

                if added_new_context and verbose:
                    print(
                        f"Retrieved and added {len(newly_retrieved_docs)} new documents for hop {num_actual_hops_taken}.")
            elif verbose:
                print(
                    f"No new documents retrieved for hop {num_actual_hops_taken} based on query: '{current_query_for_retrieval[:100]}...'")
        elif verbose and hop_iter_num < max_hops - 1:
            print(
                "Skipping retrieval for this hop (either answer indicated or no valid query).")

    if verbose:
        print(
            f"\n--- Generating Final Answer (after {num_actual_hops_taken} hop iterations) ---")
    final_answer_prompt_str = construct_final_answer_prompt(
        original_question=original_question,
        full_accumulated_context=accumulated_context_str,
        full_chain_of_thought=full_chain_of_thought_list,
        # Use dynamic context limit
        max_context_tokens_for_final_answer=max_final_context_tokens
    )
    final_answer_text = llm_wrapper.generate(
        final_answer_prompt_str, max_new_tokens=max_final_answer_tokens, temperature=0.3
    )
    if verbose:
        print(f"Final Generated Answer: {final_answer_text}")
    return final_answer_text, num_actual_hops_taken, full_chain_of_thought_list, accumulated_context_str


def run_qa_system(
    question_data: dict,  # A single item from your qa_dataset
    retriever: RetrieverType,
    llm_wrapper: LLMWrapper,
    mode: str = "multi-hop-ircot",
    max_ircot_hops: int = 3,
    k_retrieve_single_hop: int = 5,
    k_retrieve_multi_hop: int = 2,
    verbose_level: int = 1  # 0: silent, 1: summary, 2: detailed per-hop
) -> dict:
    """
    Main orchestrator to run QA in different modes and return results.
    """
    original_question = question_data['query']
    ground_truth_answer = question_data['answer']

    result = {
        "question": original_question,
        "ground_truth": ground_truth_answer,
        "mode": mode,
        "predicted_answer": None,
        "f1_score": 0.0,
        "hops_taken": 0,
        "chain_of_thought": None,
        "retrieved_context": None
    }

    is_verbose_detail = verbose_level >= 2
    is_verbose_summary = verbose_level >= 1

    if mode == "llm-only":  # LLM Only
        predicted_answer, hops = answer_question_llm_only(
            original_question, llm_wrapper, verbose=is_verbose_detail
        )
        result["predicted_answer"] = predicted_answer
        result["hops_taken"] = hops

    elif mode == "single-hop-rag":  # Single-Hop RAG
        predicted_answer, hops, context = answer_question_single_hop_rag(
            original_question, retriever, llm_wrapper,
            k_retrieve=k_retrieve_single_hop, verbose=is_verbose_detail
        )
        result["predicted_answer"] = predicted_answer
        result["hops_taken"] = hops
        result["retrieved_context"] = context

    elif mode == "multi-hop-ircot":  # Multi-Hop IR-CoT
        predicted_answer, hops, cot, context = answer_question_multi_hop_ircot(
            original_question, retriever, llm_wrapper,
            max_hops=max_ircot_hops,
            k_retrieve_per_hop=k_retrieve_multi_hop,
            verbose=is_verbose_detail
        )
        result["predicted_answer"] = predicted_answer
        # This is the number of CoT/reasoning steps
        result["hops_taken"] = hops
        result["chain_of_thought"] = cot
        result["retrieved_context"] = context

    else:
        print(
            f"Error: Unknown mode {mode}. Supported modes are 'llm-only', 'single-hop-rag', 'multi-hop-ircot'.")
        return result  # Return with Nones

    # Calculate F1 score
    if result["predicted_answer"] is not None:
        result["f1_score"] = calculate_f1_score(
            result["predicted_answer"], ground_truth_answer)

    if is_verbose_summary:
        print(f"Mode {mode} Result for Q: '{original_question}'")
        print(f"Predicted: {result['predicted_answer']}")
        print(f"Ground Truth: {result['ground_truth']}")
        print(f"F1: {result['f1_score']:.4f}, Hops: {result['hops_taken']}")

    return result
