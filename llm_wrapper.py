import os
import torch
from dotenv import load_dotenv

load_dotenv()


class LLMWrapper:
    MODEL_CONTEXT_LIMITS = {
        "gemma3:4b-it-qat": 128000,
        "qwen3:8b": 40000,
        "llama3.2:3b": 128000,
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
        "gpt-4-turbo": 128000,
        "default": 4096
    }

    def __init__(self, model_identifier, api_key=None, llm_type="generic", open_router_base_url="https://openrouter.ai/api/v1", ollama_base_url="http://localhost:11434"):
        """
        Initializes the LLM wrapper.
        Args:
            model_identifier (str): Name of the model, endpoint, or path to local model.
                                    For OpenRouter, this is like "mistralai/mistral-7b-instruct".
                                    For Ollama, this is like "llama2", "mistral", etc.
            api_key (str, optional): API key if required (e.g., for OpenAI or OpenRouter).
                                     If None, will try to load from OPENROUTER_API_KEY or OPENAI_API_KEY.
            llm_type (str): Helps in conditional logic. e.g., "openai", "huggingface_local", "open_router", "ollama", "generic".
            open_router_base_url (str): The base URL for OpenRouter API.
            ollama_base_url (str): The base URL for Ollama API (default: http://localhost:11434).
        """
        self.model_identifier = model_identifier
        self.llm_type = llm_type.lower()  # Normalize for easier comparison
        self.client = None  # Initialize client to None

        if self.llm_type == "openai":
            import openai  # Import here to avoid dependency if not used
            # Use provided api_key or fall back to environment variable
            self.api_key = api_key if api_key else os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "OpenAI API key not provided or found in OPENAI_API_KEY environment variable.")
            self.client = openai.OpenAI(api_key=self.api_key)

        elif self.llm_type == "open_router":
            import openai  # OpenRouter uses OpenAI-compatible API
            # Use provided api_key or fall back to environment variable
            self.api_key = api_key if api_key else os.getenv(
                "OPENROUTER_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "OpenRouter API key not provided or found in OPENROUTER_API_KEY environment variable.")
            self.client = openai.OpenAI(
                base_url=open_router_base_url,
                api_key=self.api_key,
            )
            print(
                f"OpenRouter client configured for model: {self.model_identifier}")

        elif self.llm_type == "ollama":
            try:
                import requests
                self.ollama_base_url = ollama_base_url
                # Test connection to Ollama
                response = requests.get(
                    f"{self.ollama_base_url}/api/tags", timeout=5)
                if response.status_code == 200:
                    available_models = [model['name']
                                        for model in response.json().get('models', [])]
                    print(
                        f"Connected to Ollama. Available models: {available_models}")
                    if self.model_identifier not in available_models:
                        print(
                            f"Warning: Model '{self.model_identifier}' not found in available models. You may need to pull it first with 'ollama pull {self.model_identifier}'")
                else:
                    print(
                        f"Warning: Could not connect to Ollama at {self.ollama_base_url}")
            except ImportError:
                print("Requests library not installed. `pip install requests`")
                self.ollama_base_url = None
            except Exception as e:
                print(f"Error connecting to Ollama: {e}")
                # Keep the URL even if connection failed
                self.ollama_base_url = ollama_base_url

        elif self.llm_type == "huggingface_local":
            from transformers import pipeline
            # Ensure you have torch if using GPU
            device = 0 if torch.cuda.is_available() else -1
            try:
                self.generator = pipeline(
                    "text-generation",
                    model=self.model_identifier,
                    device=device
                )
            except Exception as e:
                print(
                    f"Error initializing Hugging Face pipeline for {self.model_identifier}: {e}")
                self.generator = None

        else:  # Covers "generic" and any other unspecified type
            print(
                f"LLM type '{self.llm_type}' is generic or unsupported by default. No specific client initialized.")

        print(
            f"LLMWrapper initialized for model: {self.model_identifier} (type: {self.llm_type})")

    def generate(self, prompt, max_new_tokens=150, temperature=0.7, stop_sequences=None, verbose=False):
        """
        Generates text using the configured LLM.
        Args:
            prompt (str): The input prompt for the LLM.
            max_new_tokens (int): Maximum number of new tokens to generate.
            temperature (float): Controls randomness. Lower is more deterministic.
            stop_sequences (list, optional): A list of strings that, if generated, will cause generation to stop.
        Returns:
            str: The LLM's generated text.
        """
        # print(
        #     f"\n--- Sending prompt to LLM ({self.model_identifier} via {self.llm_type}) ---")
        # print(f"Prompt (first 300 chars):\n{prompt[:300]}...\n")

        # Default
        generated_text = f"LLM_RESPONSE_PLACEHOLDER_FOR_PROMPT: {prompt[:50]}..."

        if self.llm_type == "openai" or self.llm_type == "open_router":
            if not self.client:
                print(f"Error: {self.llm_type} client not initialized.")
                return f"ERROR_CLIENT_NOT_INITIALIZED_FOR_{self.llm_type.upper()}"
            try:
                # Both OpenAI and OpenRouter use this chat completions structure
                response = self.client.chat.completions.create(
                    model=self.model_identifier,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    stop=stop_sequences
                )
                generated_text = response.choices[0].message.content.strip()
            except Exception as e:
                print(
                    f"Error calling {self.llm_type} API for model {self.model_identifier}: {e}")
                generated_text = f"ERROR_CALLING_{self.llm_type.upper()}_API: {e}"

        elif self.llm_type == "ollama":
            if not hasattr(self, 'ollama_base_url') or self.ollama_base_url is None:
                print(
                    "Error: Ollama base URL not set or requests library not available.")
                return "ERROR_OLLAMA_NOT_CONFIGURED"
            try:
                import requests
                import json

                # Prepare the request payload for Ollama
                payload = {
                    "model": self.model_identifier,
                    "prompt": prompt,
                    "stream": False,  # Get complete response at once
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_new_tokens,  # Ollama uses num_predict instead of max_tokens
                    },
                    "think": False
                }

                # Add stop sequences if provided
                if stop_sequences:
                    payload["options"]["stop"] = stop_sequences

                # Make the request to Ollama
                response = requests.post(
                    f"{self.ollama_base_url}/api/generate",
                    json=payload,
                    timeout=120  # Longer timeout for generation
                )

                if response.status_code == 200:
                    result = response.json()
                    generated_text = result.get("response", "").strip()
                else:
                    print(
                        f"Ollama API returned status code {response.status_code}: {response.text}")
                    generated_text = f"ERROR_OLLAMA_API_STATUS_{response.status_code}"

            except ImportError:
                print("Requests library not installed. `pip install requests`")
                generated_text = "ERROR_REQUESTS_NOT_INSTALLED"
            except Exception as e:
                print(f"Error calling Ollama API: {e}")
                generated_text = f"ERROR_CALLING_OLLAMA_API: {e}"

        elif self.llm_type == "huggingface_local":
            if not hasattr(self, 'generator') or self.generator is None:
                print("Error: Hugging Face generator not initialized.")
                return "ERROR_HF_GENERATOR_NOT_INITIALIZED"
            try:
                # max_length = len(self.generator.tokenizer.encode(prompt)) + max_new_tokens # Alternative way to set max_length
                result = self.generator(
                    prompt,
                    max_new_tokens=max_new_tokens,  # Some pipelines prefer max_new_tokens
                    # max_length=max_length, # Some pipelines prefer max_length
                    temperature=temperature,
                    num_return_sequences=1,
                    pad_token_id=self.generator.tokenizer.eos_token_id,
                    # Explicitly set eos_token_id for stopping
                    eos_token_id=self.generator.tokenizer.eos_token_id
                )
                raw_generated_text = result[0]['generated_text']

                # Strip the prompt if it's included in the output
                if raw_generated_text.startswith(prompt):
                    generated_text = raw_generated_text[len(prompt):].strip()
                else:
                    generated_text = raw_generated_text.strip()
            except Exception as e:
                print(f"Error using Hugging Face pipeline: {e}")
                generated_text = f"ERROR_USING_HUGGINGFACE_MODEL: {e}"

        elif self.llm_type == "generic":
            print(
                "Generic LLM type: Returning placeholder response. Implement actual call.")

        else:
            print(
                f"Unsupported LLM type '{self.llm_type}' in generate method.")
            generated_text = f"ERROR_UNSUPPORTED_LLM_TYPE: {self.llm_type}"

        if verbose:
            print(
                f"LLM Raw Output (first 300 chars): {generated_text[:300]}...")
        return generated_text.strip()

    def get_context_limit(self):
        """
        Get the context limit for the current model.

        Returns:
            int: Context limit in tokens
        """
        # Check exact match first
        if self.model_identifier in self.MODEL_CONTEXT_LIMITS:
            return self.MODEL_CONTEXT_LIMITS[self.model_identifier]

        # Check partial matches for model families
        model_lower = self.model_identifier.lower()

        if "gemma" in model_lower:
            return 128000
        elif "qwen" in model_lower:
            return 40000
        elif "llama3.2" in model_lower or "llama3.1" in model_lower:
            return 128000
        elif "gpt-4" in model_lower:
            if "turbo" in model_lower:
                return 128000
            elif "32k" in model_lower:
                return 32768
            else:
                return 8192
        elif "gpt-3.5" in model_lower:
            if "16k" in model_lower:
                return 16384
            else:
                return 4096
        else:
            print(
                f"Warning: Unknown model {self.model_identifier}, using default context limit of 4096")
            return 4096

    def get_recommended_context_allocation(self):
        """
        Get recommended context allocation for different prompt components.

        Returns:
            dict: Dictionary with recommended token allocations
        """
        total_context = self.get_context_limit()

        # Reserve space for prompt structure, instructions, and output
        reserved_tokens = 2000
        available_tokens = max(1000, total_context - reserved_tokens)

        # Allocate tokens based on total available context
        if total_context >= 100000:  # Large context models (128K+)
            return {
                "cot_step_context": min(50000, int(available_tokens * 0.7)),
                "final_answer_context": min(30000, int(available_tokens * 0.5)),
                "max_accumulated_context": min(80000, int(available_tokens * 0.8))
            }
        elif total_context >= 30000:  # Medium context models (32K-40K)
            return {
                "cot_step_context": min(20000, int(available_tokens * 0.6)),
                "final_answer_context": min(15000, int(available_tokens * 0.5)),
                "max_accumulated_context": min(25000, int(available_tokens * 0.7))
            }
        elif total_context >= 8000:  # Small-medium context models
            return {
                "cot_step_context": min(5000, int(available_tokens * 0.5)),
                "final_answer_context": min(3000, int(available_tokens * 0.4)),
                "max_accumulated_context": min(6000, int(available_tokens * 0.6))
            }
        else:  # Very small context models
            return {
                "cot_step_context": min(2000, int(available_tokens * 0.4)),
                "final_answer_context": min(1500, int(available_tokens * 0.3)),
                "max_accumulated_context": min(2500, int(available_tokens * 0.5))
            }

# --- Example Usage ---

# Create a .env file in your project root with:
# OPENAI_API_KEY="sk-your-openai-key"
# OPENROUTER_API_KEY="sk-or-your-openrouter-key"

# --- OpenAI Example ---
# try:
#     llm_openai = LLMWrapper(
#         model_identifier="gpt-3.5-turbo",
#         llm_type="openai"
#         # api_key can be omitted if OPENAI_API_KEY is in .env
#     )
#     test_prompt_openai = "What is the capital of France? The capital of France is"
#     response_openai = llm_openai.generate(test_prompt_openai, max_new_tokens=10)
#     print(f"\nOpenAI Test Response: {response_openai}")
# except ValueError as e:
#     print(e)
# except ImportError:
#     print("OpenAI library not installed. `pip install openai`")


# --- OpenRouter Example ---
# try:
#     llm_openrouter = LLMWrapper(
#         model_identifier="mistralai/mistral-7b-instruct", # Choose a model available on OpenRouter
#         llm_type="open_router"
#         # api_key can be omitted if OPENROUTER_API_KEY is in .env
#     )
#     test_prompt_or = "Write a short poem about a robot learning to dream."
#     response_or = llm_openrouter.generate(test_prompt_or, max_new_tokens=50)
#     print(f"\nOpenRouter Test Response: {response_or}")
# except ValueError as e:
#     print(e)
# except ImportError:
#     print("OpenAI library (used for OpenRouter) not installed. `pip install openai`")


# --- Ollama Local Example ---
# try:
#     # Make sure you have Ollama running locally and a model pulled (e.g., `ollama pull llama2`)
#     llm_ollama = LLMWrapper(
#         model_identifier="llama2",  # or "mistral", "codellama", etc.
#         llm_type="ollama"
#         # ollama_base_url can be omitted if using default localhost:11434
#     )
#     test_prompt_ollama = "Explain the concept of recursion in programming."
#     response_ollama = llm_ollama.generate(test_prompt_ollama, max_new_tokens=100)
#     print(f"\nOllama Test Response: {response_ollama}")
# except ImportError:
#     print("Requests library not installed. `pip install requests`")


# --- Hugging Face Local Example ---
# try:
#     # Make sure you have a model like "gpt2" downloaded or accessible
#     llm_hf = LLMWrapper(
#         model_identifier="gpt2",
#         llm_type="huggingface_local"
#     )
#     if hasattr(llm_hf, 'generator') and llm_hf.generator: # Check if generator was initialized
#         test_prompt_hf = "Once upon a time, in a land far away,"
#         response_hf = llm_hf.generate(test_prompt_hf, max_new_tokens=20)
#         print(f"\nHugging Face Test Response: {response_hf}")
#     else:
#         print("Skipping Hugging Face test as generator was not initialized.")
# except ImportError:
#     print("Transformers or PyTorch library not installed. `pip install transformers torch`")


# --- Generic Placeholder Example ---
# llm_generic = LLMWrapper(model_identifier="test_model", llm_type="generic")
# test_prompt_generic = "This is a test prompt for the generic LLM."
# response_generic = llm_generic.generate(test_prompt_generic)
# print(f"\nGeneric Test Response: {response_generic}")
