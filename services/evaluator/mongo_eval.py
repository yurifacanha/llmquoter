from pymongo import MongoClient

from core.metrics import bm25_score, f1_score
from core.quote_utils import format_score
from services.evaluator.llm_eval import (
    evaluate_single_model,
    compute_aggregate_metrics,
    compute_bm25_aggregate,
)


def evaluate_from_llmquoter_test(
    models: list[str] | None = None,
    uuid: str | None = None,
    force: bool = False,
    verbose: bool | None = None,
    max_workers: int = 5,
    collection_name: str = "LLMQuoterTest",
    connection_string: str = "mongodb://localhost:27017",
    database_name: str = "llmquoter",
) -> dict:
    client = MongoClient(connection_string)
    db = client[database_name]
    collection = db[collection_name]
    filter_query = {"quotes": {"$exists": True, "$ne": ""}, "inferences": {"$exists": True}}
    if uuid:
        filter_query["uuid"] = uuid
        print(f"Filtering to single document: {uuid}")
    docs = list(collection.find(filter_query))
    client.close()
    print(f"Loaded {len(docs)} documents from {collection_name}")
    if not docs:
        return {"models": {}, "count": 0}
    all_models = set()
    for doc in docs:
        all_models.update(doc.get("inferences", {}).keys())
    models_to_eval = sorted(all_models) if not models else [m for m in models if m in all_models]
    models_to_eval = [m for m in models_to_eval if m in all_models]
    print(f"Models to evaluate: {models_to_eval}")
    verbose_prompts = verbose if verbose is not None else bool(uuid)
    for model_name in models_to_eval:
        print(f"\n=== Evaluating {model_name} ===")
        samples = []
        for doc in docs:
            uuid_val = doc.get("uuid")
            if not uuid_val:
                continue
            if not force and doc.get("scores", {}).get(model_name):
                continue
            inference = doc.get("inferences", {}).get(model_name)
            if not inference:
                continue
            ground_truth = doc.get("quotes", "")
            if not ground_truth:
                continue
            samples.append({
                "uuid": uuid_val,
                "ground_truth": ground_truth,
                "system_response": inference
            })
        if not samples:
            print("No samples to evaluate (all already scored)")
            continue
        print(f"Evaluating {len(samples)} samples...")
        results = evaluate_single_model(
            samples,
            max_workers=max_workers,
            model_name=model_name,
            save_to_mongo=True,
            collection_name=collection_name,
            id_field="uuid",
            verbose=verbose_prompts,
        )
        llm_metrics = compute_aggregate_metrics(results)
        bm25_metrics = compute_bm25_aggregate(results)
        format_scores = [r.get("format_score", 0.0) for r in results if r.get("format_score") is not None]
        avg_format = round(sum(format_scores) / len(format_scores), 4) if format_scores else 0.0
        print(f"{model_name}: R={llm_metrics['avg_recall']:.4f} P={llm_metrics['avg_precision']:.4f} F1={llm_metrics['avg_f1']:.4f} BM25={bm25_metrics['avg_bm25']:.4f} FMT={avg_format:.4f}")
    return {"models": models_to_eval, "count": len(docs)}


def update_scores_manually(
    manual_scores: list[dict],
    collection_name: str = "LLMQuoterTest",
    connection_string: str = "mongodb://localhost:27017",
    database_name: str = "llmquoter",
) -> int:
    client = MongoClient(connection_string)
    collection = client[database_name][collection_name]
    updated = 0
    for item in manual_scores:
        uuid_val = item.get("uuid")
        model_name = item.get("model")
        recall = item.get("recall", 0.0)
        precision = item.get("precision", 0.0)
        if not uuid_val or not model_name:
            continue
        doc = collection.find_one({"uuid": uuid_val})
        if not doc:
            continue
        ground_truth = doc.get("quotes", "")
        system_response = doc.get("inferences", {}).get(model_name)
        if not system_response:
            continue
        bm25 = bm25_score(ground_truth, system_response)
        fmt = format_score(system_response)
        f1 = f1_score(precision, recall)
        score_result = {
            "recall": round(float(recall), 4),
            "precision": round(float(precision), 4),
            "f1": round(f1, 4),
            "bm25": bm25,
            "format_score": fmt
        }
        result = collection.update_one(
            {"uuid": uuid_val},
            {"$set": {f"scores.{model_name}": score_result}}
        )
        if result.modified_count > 0 or result.matched_count > 0:
            updated += 1
            print(f"Manual update {uuid_val[:8]}... {model_name} [R:{recall} P:{precision} F1:{f1:.3f} BM25:{bm25} FMT:{fmt}]")
    client.close()
    print(f"Updated {updated} documents with manual scores")
    return updated
