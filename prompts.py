def construct_cot_step_prompt(original_question: str,
                              accumulated_context: str,
                              cot_so_far: list[str],
                              max_context_tokens=35000  # Approximate token limit for context
                              ) -> str:
    """
    Constructs the prompt for the LLM to generate the next CoT step.

    Args:
        original_question (str): The initial multi-hop question.
        accumulated_context (str): Text from all retrieved passages so far.
        cot_so_far (list[str]): A list of CoT sentences generated in previous steps.
        max_context_tokens (int): An approximate limit for the context part of the prompt
                                  to avoid exceeding LLM input limits. Simple character count for now.

    Returns:
        str: The formatted prompt string.
    """

    # Context Management (truncate if too long)
    reserved_chars_for_other_parts = len(original_question) + \
        len("\n".join(cot_so_far)) + \
        500  # For instructions and formatting

    max_chars_for_context_in_prompt = (
        max_context_tokens * 3) - reserved_chars_for_other_parts

    if len(accumulated_context) > max_chars_for_context_in_prompt:
        print(
            f"Warning: Truncating accumulated context from {len(accumulated_context)} to {max_chars_for_context_in_prompt} chars for CoT prompt.")
        context_to_include = accumulated_context[:max_chars_for_context_in_prompt] + \
            "...\n[Context Truncated]"
    else:
        context_to_include = accumulated_context

    # Constructing the Chain of Thought so far
    if cot_so_far:
        reasoning_history = "Previous reasoning steps:\n" + \
            "\n".join(f"- {step}" for step in cot_so_far)
    else:
        reasoning_history = "This is the first reasoning step."

    # Prompt Template: It needs to be iterative. At each step, we're asking for the *next* thought.

    prompt = f"""You are a helpful assistant performing multi-step reasoning to answer a complex question.
                You will be given a question, some retrieved context, and any previous reasoning steps.
                Your task is to generate the *single next logical reasoning step* based on the available information to help answer the question.
                Focus on what information is still missing or what connection needs to be made.

                If you believe you have enough information from the context and previous reasoning to directly and confidently answer the original question,
                your response should start *ONLY* with the phrase "The answer is: " followed by the answer. Do not add any other prefix.

                Otherwise, provide only the single next reasoning sentence that helps progress towards the answer. Do not try to answer the question prematurely if you are still reasoning.

                Original Question: {original_question}

                Retrieved Context:
                ---
                {context_to_include}
                ---

                {reasoning_history}

                Based on the original question, the retrieved context, and the previous reasoning steps, what is the single next reasoning step or the final answer (if known)?
                Tip: The reasoning is a single sentences that tell you expected to be the missing information to answer the question.
                """

    return prompt


def construct_final_answer_prompt(original_question: str,
                                  full_accumulated_context: str,
                                  full_chain_of_thought: list[str],
                                  max_context_tokens_for_final_answer=2000  # Approx token limit for context
                                  ) -> str:
    """
    Constructs the prompt for the LLM to generate the final answer
    based on all gathered information (Direct Prompting style).

    Args:
        original_question (str): The initial multi-hop question.
        full_accumulated_context (str): Text from all retrieved passages from all hops.
        full_chain_of_thought (list[str]): All CoT sentences generated during the IR-CoT loop.
        max_context_tokens_for_final_answer (int): Approximate token limit for the context
                                                   and CoT part of the prompt.

    Returns:
        str: The formatted prompt string.
    """

    # Context and CoT Management (Simple version: truncate if too long)

    combined_evidence_text = "Reasoning Steps Taken:\n" + "\n".join(f"- {step}" for step in full_chain_of_thought) + \
                             "\n\nSupporting Retrieved Context:\n---\n" + full_accumulated_context + "\n---"

    # Rough character-based truncation
    # Reserve space for question and instructions
    reserved_chars_for_other_parts_final = len(
        original_question) + 300  # For instructions

    max_chars_for_evidence_in_prompt = (
        max_context_tokens_for_final_answer * 3) - reserved_chars_for_other_parts_final

    if len(combined_evidence_text) > max_chars_for_evidence_in_prompt:
        print(
            f"Warning: Truncating combined evidence from {len(combined_evidence_text)} to {max_chars_for_evidence_in_prompt} chars for final answer prompt.")
        evidence_to_include = combined_evidence_text[:max_chars_for_evidence_in_prompt] + \
            "...\n[Evidence Truncated]"
    else:
        evidence_to_include = combined_evidence_text

    # Prompt Template (Direct Answer Style)
    prompt = f"""You are an intelligent assistant. Based on the following reasoning steps and supporting context,
                please provide a concise and direct answer to the original question.

                Original Question: {original_question}

                {evidence_to_include}

                Based on all the information above, what is the final answer to the original question?
                DIRECT ANSWER ONLY NO EXPLANATION('Yes','No', Name, Place, Number, 'Insufficient information', etc.)
                The Final Answer for {original_question} is: """

    return prompt
