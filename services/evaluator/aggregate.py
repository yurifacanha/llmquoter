from pymongo import MongoClient


def get_model_averages_llmquoter_test(
    model_name: str | None,
    collection_name: str,
    connection_string: str,
    database_name: str,
) -> dict:
    client = MongoClient(connection_string)
    db = client[database_name]
    collection = db[collection_name]
    docs = list(collection.find({"scores": {"$exists": True}}, {"scores": 1}))
    client.close()
    if not docs:
        return {}
    model_results = {}
    for doc in docs:
        scores = doc.get("scores", {})
        if not scores:
            continue
        if model_name:
            if model_name in scores:
                if model_name not in model_results:
                    model_results[model_name] = []
                model_results[model_name].append(scores[model_name])
        else:
            for mod_name, metrics in scores.items():
                if mod_name not in model_results:
                    model_results[mod_name] = []
                model_results[mod_name].append(metrics)
    averages = {}
    for mod_name, metrics_list in model_results.items():
        if not metrics_list:
            continue
        recalls = [max(0.0, m.get("recall") or 0.0) for m in metrics_list]
        precisions = [max(0.0, m.get("precision") or 0.0) for m in metrics_list]
        f1s = [max(0.0, m.get("f1") or 0.0) for m in metrics_list]
        bm25s = [max(0.0, m.get("bm25") or 0.0) for m in metrics_list]
        format_scores = [max(0.0, m.get("format_score") or 0.0) for m in metrics_list]
        averages[mod_name] = {
            "avg_recall": round(sum(recalls) / len(recalls), 4) if recalls else 0.0,
            "avg_precision": round(sum(precisions) / len(precisions), 4) if precisions else 0.0,
            "avg_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
            "avg_bm25": round(sum(bm25s) / len(bm25s), 4) if bm25s else 0.0,
            "avg_format_score": round(sum(format_scores) / len(format_scores), 4) if format_scores else 0.0,
            "count": len(metrics_list)
        }
    return averages


def print_model_averages_llmquoter_test(
    model_name: str | None = None,
    collection_name: str = "LLMQuoterTest",
) -> dict:
    averages = get_model_averages_llmquoter_test(
        model_name=model_name,
        collection_name=collection_name,
        connection_string="mongodb://localhost:27017",
        database_name="llmquoter",
    )
    if not averages:
        print("No results found.")
        return {}
    if model_name:
        if model_name in averages:
            m = averages[model_name]
            print(f"\n=== {model_name} (LLMQuoterTest) ===")
            print(f"Count: {m['count']} | Recall: {m['avg_recall']:.4f} | Precision: {m['avg_precision']:.4f} | F1: {m['avg_f1']:.4f} | BM25: {m['avg_bm25']:.4f} | Format: {m['avg_format_score']:.4f}")
        else:
            print(f"No results for {model_name}")
    else:
        print("\n=== LLMQuoterTest Average Scores ===")
        print(f"{'Model':<20} {'Count':<8} {'Recall':<10} {'Precision':<10} {'F1':<10} {'BM25':<10} {'Format':<10}")
        print("-" * 90)
        for mod_name, m in sorted(averages.items()):
            print(f"{mod_name:<20} {m['count']:<8} {m['avg_recall']:<10.4f} {m['avg_precision']:<10.4f} {m['avg_f1']:<10.4f} {m['avg_bm25']:<10.4f} {m['avg_format_score']:<10.4f}")
    return averages
