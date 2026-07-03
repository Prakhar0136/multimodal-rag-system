# test_evaluation.py
import json
from src.main_pipeline import MultimodalRAGPipeline
from src.evaluation.evaluator import PipelineEvaluator

def run_evaluation_suite():
    print("🛠️ Initializing Evaluation Harness...")
    pipeline = MultimodalRAGPipeline()
    evaluator = PipelineEvaluator()

    # Load dataset
    with open("data/eval_dataset.json", "r") as f:
        dataset = json.load(f)

    results = []
    total_scores = {"faithfulness": 0, "relevance": 0, "context_recall": 0}

    print(f"🏃 Running {len(dataset)} test cases...\n")

    for item in dataset:
        print(f"Testing Query: {item['id']} - {item['type']}")
        
        # 1. Execute Pipeline
        pipeline_output = pipeline.execute_query(item["query"], user_id="eval_harness")
        
        # 2. Grade Pipeline
        scores = evaluator.evaluate_turn(
            query=item["query"],
            context_used=pipeline_output.get("raw_context_used", []),
            generated_answer=pipeline_output["response"],
            expected_facts=item["expected_facts"]
        )
        
        # Aggregate scores
        for k in total_scores.keys():
            total_scores[k] += scores.get(k, 0)
            
        results.append({
            "id": item["id"],
            "query": item["query"],
            "scores": scores
        })

    # 3. Generate Markdown Report
    generate_markdown_report(results, total_scores, len(dataset))

def generate_markdown_report(results, total_scores, total_tests):
    report = "# 📊 Production RAG Architecture Evaluation Report\n\n"
    report += "## Aggregate Metrics\n"
    report += f"- **Faithfulness (Hallucination Mitigation):** {(total_scores['faithfulness'] / total_tests) * 100:.1f}%\n"
    report += f"- **Answer Relevance:** {(total_scores['relevance'] / total_tests) * 100:.1f}%\n"
    report += f"- **Context Recall (Retrieval Accuracy):** {(total_scores['context_recall'] / total_tests) * 100:.1f}%\n\n"
    
    report += "## Per-Query Breakdown\n"
    report += "| Test ID | Faithfulness | Relevance | Recall |\n"
    report += "| :--- | :--- | :--- | :--- |\n"
    
    for res in results:
        report += f"| {res['id']} | {res['scores'].get('faithfulness', 0)} | {res['scores'].get('relevance', 0)} | {res['scores'].get('context_recall', 0)} |\n"
        
    with open("data/eval_report.md", "w", encoding="utf-8") as f:
        f.write(report)
        
    print("\n✅ Evaluation complete. Markdown report generated at 'data/eval_report.md'.")

if __name__ == "__main__":
    run_evaluation_suite()