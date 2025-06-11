import os
import json
import time
from datetime import datetime
from pathlib import Path

from utils import load_corpus, load_summary_corpus, load_queries_and_answers
from retriever import BM25Retriever, SummaryBM25Retriever
from llm_wrapper import LLMWrapper
from evaluation import run_comprehensive_evaluation, print_evaluation_summary


class ExperimentRunner:
    def __init__(self, results_dir="experiment_results"):
        """
        Initialize the experiment runner.

        Args:
            results_dir: Directory to save all results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        # Create timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.results_dir / f"run_{self.timestamp}"
        self.run_dir.mkdir(exist_ok=True)

        print(f"📁 Results will be saved to: {self.run_dir}")

        # Configuration
        self.config = {
            "corpus_filepath": "data/multihoprag_corpus.txt",
            "summary_corpus_filepath": "data_summary/multihoprag_corpus_summary.csv",
            "qa_filepath": "data/MultiHopRAG.json",
            "num_samples": 200,
            "max_ircot_hops": 10,
            "k_retrieve_multi_hop": 2,
            "k_retrieve_single_hop": 2,
            "similarity_threshold": 0.7,
            "verbose_level": 0
        }

        # LLM configurations to test
        self.llm_configs = [
            {"model_identifier": "gemma3:4b-it-qat",
                "llm_type": "ollama", "name": "Gemma3_4B"},
            {"model_identifier": "qwen3:8b",
                "llm_type": "ollama", "name": "Qwen3_8B"},
            {"model_identifier": "llama3.2:3b",
                "llm_type": "ollama", "name": "Llama3_2_3B"}
        ]

        # Modes to test
        self.modes = ["multi-hop-ircot", "single-hop-rag", "llm-only"]

        # Corpus types to test
        self.corpus_types = ["original", "summary"]

        # Initialize results tracking
        self.all_results = {}

    def save_config(self):
        """Save experiment configuration."""
        config_file = self.run_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump({
                "timestamp": self.timestamp,
                "config": self.config,
                "llm_configs": self.llm_configs,
                "modes": self.modes,
                "corpus_types": self.corpus_types
            }, f, indent=2)
        print(f"✅ Configuration saved to {config_file}")

    def load_datasets(self):
        """Load all required datasets."""
        print("🔄 Loading datasets...")

        self.original_corpus = load_corpus(self.config["corpus_filepath"])
        self.summary_corpus = load_summary_corpus(
            self.config["summary_corpus_filepath"])
        self.qa_dataset = load_queries_and_answers(self.config["qa_filepath"])

        if not self.original_corpus or not self.summary_corpus or not self.qa_dataset:
            raise ValueError(
                "Failed to load one or more datasets. Please check file paths.")

        print(f"✅ Loaded {len(self.original_corpus)} original documents")
        print(f"✅ Loaded {len(self.summary_corpus)} summary documents")
        print(f"✅ Loaded {len(self.qa_dataset)} questions")

        # Save dataset info
        dataset_info = {
            "original_corpus_size": len(self.original_corpus),
            "summary_corpus_size": len(self.summary_corpus),
            "qa_dataset_size": len(self.qa_dataset),
            "num_samples_tested": self.config["num_samples"]
        }

        with open(self.run_dir / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)

    def initialize_retrievers(self):
        """Initialize retrievers for both corpus types."""
        print("🔄 Initializing retrievers...")

        self.original_retriever = BM25Retriever(self.original_corpus)
        self.summary_retriever = SummaryBM25Retriever(self.summary_corpus)

        print("✅ Original BM25 retriever initialized")
        print("✅ Summary BM25 retriever initialized")

    def run_single_experiment(self, llm_config, mode, corpus_type):
        """
        Run a single experiment configuration.

        Args:
            llm_config: LLM configuration dictionary
            mode: Evaluation mode (multi-hop-ircot, single-hop-rag, llm-only)
            corpus_type: Corpus type (original or summary)

        Returns:
            Tuple of (results, summary, experiment_info)
        """
        experiment_name = f"{llm_config['name']}_{mode}_{corpus_type}"
        print(f"\n🚀 Running experiment: {experiment_name}")

        # Initialize LLM
        try:
            llm = LLMWrapper(
                model_identifier=llm_config["model_identifier"],
                llm_type=llm_config["llm_type"]
            )

            # Display context information
            context_limit = llm.get_context_limit()
            context_allocation = llm.get_recommended_context_allocation()
            print(f"✅ LLM {llm_config['name']} initialized")
            print(f"   📏 Context limit: {context_limit:,} tokens")
            print(
                f"   🧠 CoT context allocation: {context_allocation['cot_step_context']:,} tokens")
            print(
                f"   📝 Final answer context allocation: {context_allocation['final_answer_context']:,} tokens")
            print(
                f"   📚 Max accumulated context: {context_allocation['max_accumulated_context']:,} tokens")

        except Exception as e:
            print(f"❌ Failed to initialize LLM {llm_config['name']}: {e}")
            return None, None, {"error": str(e)}

        # Select retriever
        retriever = self.original_retriever if corpus_type == "original" else self.summary_retriever

        # Record start time
        start_time = time.time()

        try:
            # Run evaluation
            results, summary = run_comprehensive_evaluation(
                qa_dataset=self.qa_dataset,
                retriever=retriever,
                llm_wrapper=llm,
                num_samples=self.config["num_samples"],
                similarity_threshold=self.config["similarity_threshold"],
                max_ircot_hops=self.config["max_ircot_hops"],
                k_retrieve_single_hop=self.config["k_retrieve_single_hop"],
                k_retrieve_multi_hop=self.config["k_retrieve_multi_hop"],
                run_mode=mode,
                verbose_level=self.config["verbose_level"],
                verbose=False  # Reduce noise
            )

            # Record end time
            end_time = time.time()
            duration = end_time - start_time

            # Add experiment metadata
            experiment_info = {
                "experiment_name": experiment_name,
                "llm_config": llm_config,
                "mode": mode,
                "corpus_type": corpus_type,
                "duration_seconds": duration,
                "timestamp": datetime.now().isoformat(),
                "success": True,
                "context_info": {
                    "context_limit": context_limit,
                    "context_allocation": context_allocation
                }
            }

            print(
                f"✅ Experiment {experiment_name} completed in {duration:.2f} seconds")

            return results, summary, experiment_info

        except Exception as e:
            print(f"❌ Experiment {experiment_name} failed: {e}")
            return None, None, {
                "experiment_name": experiment_name,
                "error": str(e),
                "success": False
            }

    def save_experiment_results(self, experiment_name, results, summary, experiment_info):
        """Save results for a single experiment."""
        exp_dir = self.run_dir / experiment_name
        exp_dir.mkdir(exist_ok=True)

        # Save detailed results
        with open(exp_dir / "detailed_results.json", 'w') as f:
            json.dump(results, f, indent=2)

        # Save summary
        with open(exp_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        # Save experiment info
        with open(exp_dir / "experiment_info.json", 'w') as f:
            json.dump(experiment_info, f, indent=2)

        # Save human-readable summary
        with open(exp_dir / "summary.txt", 'w') as f:
            f.write(f"Experiment: {experiment_name}\n")
            f.write(
                f"Timestamp: {experiment_info.get('timestamp', 'Unknown')}\n")
            f.write(
                f"Duration: {experiment_info.get('duration_seconds', 0):.2f} seconds\n")
            f.write(
                f"LLM: {experiment_info.get('llm_config', {}).get('name', 'Unknown')}\n")
            f.write(f"Mode: {experiment_info.get('mode', 'Unknown')}\n")
            f.write(
                f"Corpus: {experiment_info.get('corpus_type', 'Unknown')}\n")

            # Add context information if available
            context_info = experiment_info.get('context_info', {})
            if context_info:
                f.write(
                    f"Context Limit: {context_info.get('context_limit', 'Unknown'):,} tokens\n")
                context_allocation = context_info.get('context_allocation', {})
                if context_allocation:
                    f.write(
                        f"CoT Context Allocation: {context_allocation.get('cot_step_context', 'Unknown'):,} tokens\n")
                    f.write(
                        f"Final Answer Context Allocation: {context_allocation.get('final_answer_context', 'Unknown'):,} tokens\n")

            f.write("="*60 + "\n")
            f.write("RESULTS SUMMARY\n")
            f.write("="*60 + "\n")
            f.write(f"Samples Evaluated: {summary.get('num_samples', 0)}\n")
            f.write(
                f"Average Answer F1: {summary.get('avg_answer_f1', 0):.4f}\n")
            f.write(f"Average Hops: {summary.get('avg_hops', 0):.2f}\n")

            if summary.get('run_mode') != 'llm-only' and summary.get('total_retrieved', 0) > 0:
                f.write(
                    f"Average Retrieval Precision: {summary.get('avg_retrieval_precision', 0):.4f}\n")
                f.write(
                    f"Average Retrieval Recall: {summary.get('avg_retrieval_recall', 0):.4f}\n")
                f.write(
                    f"Average Retrieval F1: {summary.get('avg_retrieval_f1', 0):.4f}\n")
                f.write(
                    f"Total Documents Retrieved: {summary.get('total_retrieved', 0)}\n")
                f.write(
                    f"Total Ground Truth Documents: {summary.get('total_ground_truth', 0)}\n")
                f.write(
                    f"Total Exact Matches: {summary.get('total_exact_matches', 0)}\n")
                f.write(
                    f"Total Fuzzy Matches: {summary.get('total_fuzzy_matches', 0)}\n")
                if summary.get('total_retrieved', 0) > 0:
                    match_rate = (summary.get('total_exact_matches', 0) + summary.get(
                        'total_fuzzy_matches', 0)) / summary.get('total_retrieved', 1)
                    f.write(f"Overall Match Rate: {match_rate:.4f}\n")
            else:
                f.write("Retrieval Performance: N/A (LLM-only mode)\n")

        print(f"💾 Results saved to {exp_dir}")

    def generate_comparison_report(self):
        """Generate a comprehensive comparison report."""
        print("📊 Generating comparison report...")

        comparison_file = self.run_dir / "comparison_report.txt"

        with open(comparison_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("COMPREHENSIVE EXPERIMENT COMPARISON REPORT\n")
            f.write("="*80 + "\n")
            f.write(
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Experiments: {len(self.all_results)}\n")
            f.write(f"Samples per Experiment: {self.config['num_samples']}\n")
            f.write("\n")

            # Group results by mode
            for mode in self.modes:
                f.write(f"\n{'='*60}\n")
                f.write(f"MODE: {mode.upper()}\n")
                f.write(f"{'='*60}\n")

                mode_results = {k: v for k,
                                v in self.all_results.items() if mode in k}

                if not mode_results:
                    f.write("No results for this mode.\n")
                    continue

                # Create comparison table
                f.write(
                    f"{'Experiment':<30} {'Answer F1':<12} {'Avg Hops':<10} {'Retr F1':<10} {'Context':<10} {'Duration':<10}\n")
                f.write("-" * 85 + "\n")

                for exp_name, exp_data in mode_results.items():
                    summary = exp_data['summary']
                    info = exp_data['experiment_info']

                    answer_f1 = summary.get('avg_answer_f1', 0)
                    avg_hops = summary.get('avg_hops', 0)
                    retr_f1 = summary.get(
                        'avg_retrieval_f1', 0) if mode != 'llm-only' else 0
                    duration = info.get('duration_seconds', 0)
                    context_limit = info.get('context_info', {}).get(
                        'context_limit', 'Unknown')
                    context_str = f"{context_limit//1000}K" if isinstance(
                        context_limit, int) else "N/A"

                    f.write(
                        f"{exp_name:<30} {answer_f1:<12.4f} {avg_hops:<10.2f} {retr_f1:<10.4f} {context_str:<10} {duration:<10.1f}\n")

            # Best performers
            f.write(f"\n{'='*60}\n")
            f.write("BEST PERFORMERS\n")
            f.write(f"{'='*60}\n")

            # Find best answer F1 for each mode
            for mode in self.modes:
                mode_results = {k: v for k,
                                v in self.all_results.items() if mode in k}
                if mode_results:
                    best_exp = max(mode_results.items(),
                                   key=lambda x: x[1]['summary'].get('avg_answer_f1', 0))
                    f.write(f"\nBest {mode} (Answer F1): {best_exp[0]}\n")
                    f.write(
                        f"  Answer F1: {best_exp[1]['summary'].get('avg_answer_f1', 0):.4f}\n")
                    f.write(
                        f"  Duration: {best_exp[1]['experiment_info'].get('duration_seconds', 0):.1f}s\n")

            # Model context information summary
            f.write(f"\n{'='*60}\n")
            f.write("MODEL CONTEXT INFORMATION\n")
            f.write(f"{'='*60}\n")

            # Get unique models from experiments
            models_seen = set()
            for exp_name, exp_data in self.all_results.items():
                llm_config = exp_data['experiment_info'].get('llm_config', {})
                model_name = llm_config.get('name', 'Unknown')
                model_identifier = llm_config.get(
                    'model_identifier', 'Unknown')
                context_info = exp_data['experiment_info'].get(
                    'context_info', {})

                if model_name not in models_seen:
                    models_seen.add(model_name)
                    f.write(f"\n{model_name} ({model_identifier}):\n")
                    if context_info:
                        context_limit = context_info.get(
                            'context_limit', 'Unknown')
                        context_allocation = context_info.get(
                            'context_allocation', {})
                        f.write(
                            f"  Total Context Limit: {context_limit:,} tokens\n")
                        if context_allocation:
                            f.write(
                                f"  CoT Context Allocation: {context_allocation.get('cot_step_context', 'Unknown'):,} tokens\n")
                            f.write(
                                f"  Final Answer Context: {context_allocation.get('final_answer_context', 'Unknown'):,} tokens\n")
                            f.write(
                                f"  Max Accumulated Context: {context_allocation.get('max_accumulated_context', 'Unknown'):,} tokens\n")
                    else:
                        f.write(f"  Context information not available\n")

        print(f"📋 Comparison report saved to {comparison_file}")

    def run_all_experiments(self):
        """Run all experiment combinations."""
        print("🏃‍♂️ Starting comprehensive experiment run...")

        total_experiments = len(self.llm_configs) * \
            len(self.modes) * len(self.corpus_types)
        current_experiment = 0

        for llm_config in self.llm_configs:
            for corpus_type in self.corpus_types:
                for mode in self.modes:
                    current_experiment += 1
                    experiment_name = f"{llm_config['name']}_{mode}_{corpus_type}"

                    print(f"\n{'='*60}")
                    print(
                        f"EXPERIMENT {current_experiment}/{total_experiments}: {experiment_name}")
                    print(f"{'='*60}")

                    results, summary, experiment_info = self.run_single_experiment(
                        llm_config, mode, corpus_type
                    )

                    if results is not None and summary is not None:
                        # Save results
                        self.save_experiment_results(
                            experiment_name, results, summary, experiment_info)

                        # Store in memory for comparison
                        self.all_results[experiment_name] = {
                            'results': results,
                            'summary': summary,
                            'experiment_info': experiment_info
                        }

                        # Print quick summary
                        print(f"📊 Quick Summary:")
                        print(
                            f"   Answer F1: {summary.get('avg_answer_f1', 0):.4f}")
                        print(f"   Avg Hops: {summary.get('avg_hops', 0):.2f}")
                        if mode != 'llm-only':
                            print(
                                f"   Retrieval F1: {summary.get('avg_retrieval_f1', 0):.4f}")
                    else:
                        print(f"❌ Experiment {experiment_name} failed")
                        # Still save the error info
                        error_dir = self.run_dir / experiment_name
                        error_dir.mkdir(exist_ok=True)
                        with open(error_dir / "error.json", 'w') as f:
                            json.dump(experiment_info, f, indent=2)

        # Generate final comparison report
        self.generate_comparison_report()

        print(f"\n{'='*80}")
        print("🎉 ALL EXPERIMENTS COMPLETED!")
        print(f"📁 Results saved to: {self.run_dir}")
        print(f"📊 Check comparison_report.txt for detailed analysis")
        print(f"{'='*80}")

    def run_quick_test(self, num_samples=5):
        """Run a quick test with fewer samples."""
        print(f"🧪 Running quick test with {num_samples} samples...")

        # Temporarily modify config
        original_samples = self.config["num_samples"]
        self.config["num_samples"] = num_samples

        # Run with first LLM and original corpus only
        llm_config = self.llm_configs[0]

        for mode in self.modes:
            experiment_name = f"QUICKTEST_{llm_config['name']}_{mode}_original"
            print(f"\n🧪 Quick test: {experiment_name}")

            results, summary, experiment_info = self.run_single_experiment(
                llm_config, mode, "original"
            )

            if results is not None:
                print(f"✅ Answer F1: {summary.get('avg_answer_f1', 0):.4f}")
            else:
                print(f"❌ Quick test failed")

        # Restore original config
        self.config["num_samples"] = original_samples
        print("🧪 Quick test completed!")


def main():
    """Main function to run experiments."""
    print("🚀 Starting IR-CoT Comprehensive Experiment Runner")
    print("="*60)

    # Initialize experiment runner
    runner = ExperimentRunner()

    # Save configuration
    runner.save_config()

    # Load datasets
    runner.load_datasets()

    # Initialize retrievers
    runner.initialize_retrievers()

    # Option to run quick test first
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--quick-test":
        runner.run_quick_test()
        return

    # Run all experiments
    runner.run_all_experiments()


if __name__ == "__main__":
    main()
