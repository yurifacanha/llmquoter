import traceback
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
from pymongo import MongoClient

from core.quote_utils import parse_quotes, format_score
from core.metrics import bm25_score, f1_score
from ai.chains.evaluator import get_chain


def _fix_inconsistent_recall_precision(result: dict, ground_truth: str, system_response: str) -> dict:
    recall = result.get("recall", 0) or 0
    precision = result.get("precision", 0) or 0
    gt_quotes = parse_quotes(ground_truth)
    sys_quotes = parse_quotes(system_response)
    if recall > 0 and precision == 0 and gt_quotes and sys_quotes:
        n_gt, n_sys = len(gt_quotes), len(sys_quotes)
        result["precision"] = round(max(1.0 / n_sys, recall * n_gt / n_sys), 4)
    return result


def evaluate_single(ground_truth: str, system_response: str, chain, verbose: bool = False) -> dict | None:
    try:
        if verbose:
            system_part = "You are an expert evaluator that compares two sets of quotes and calculates semantic recall and precision.\n\nUse the SAME matching rule for both metrics: two quotes match if they convey the same main information, even if wording differs.\nMatching is symmetric: if quote A matches quote B, then B matches A.\n\n- Recall: Fraction (0.0 to 1.0) of ground truth quotes that have at least one matching quote in the system response\n- Precision: Fraction (0.0 to 1.0) of system response quotes that have at least one matching quote in the ground truth\n\nIMPORTANT: If recall > 0, then precision must also be > 0 (a matched ground truth quote implies a matching system quote)."
            user_part = f"**Ground Truth Quotes:**\n{ground_truth}\n\n**System Response Quotes:**\n{system_response}"
            print("\n=== SYSTEM PROMPT ===")
            print(system_part)
            print("\n=== USER PROMPT ===")
            print(user_part)
            print("=== END PROMPTS ===\n")
        result = chain.invoke({
            "ground_truth": ground_truth,
            "system_response": system_response
        })
        result = result.model_dump()
        result = _fix_inconsistent_recall_precision(result, ground_truth, system_response)
        return result
    except Exception as e:
        print(f"Error evaluating: {e}")
        return None


def compute_aggregate_metrics(results: list[dict], key: str = None) -> dict:
    if key:
        metrics = [r[key] for r in results if r.get(key) and r[key].get("f1") is not None]
    else:
        metrics = [r for r in results if r.get("f1") is not None]
    if not metrics:
        return {"avg_recall": 0, "avg_precision": 0, "avg_f1": 0, "count": 0}
    avg_recall = sum(m["recall"] for m in metrics) / len(metrics)
    avg_precision = sum(m["precision"] for m in metrics) / len(metrics)
    avg_f1 = sum(m["f1"] for m in metrics) / len(metrics)
    return {
        "avg_recall": round(avg_recall, 4),
        "avg_precision": round(avg_precision, 4),
        "avg_f1": round(avg_f1, 4),
        "count": len(metrics)
    }


def compute_bm25_aggregate(results: list[dict]) -> dict:
    scores = [r.get("bm25") or r.get("bm25_score") for r in results]
    scores = [s for s in scores if s is not None]
    if not scores:
        return {"avg_bm25": 0, "count": 0}
    return {
        "avg_bm25": round(sum(scores) / len(scores), 4),
        "count": len(scores)
    }


def evaluate_single_model(
    samples: list[dict],
    max_workers: int,
    model_name: str | None,
    save_to_mongo: bool,
    collection_name: str,
    id_field: str,
    verbose: bool,
) -> list[dict]:
    mongo_client = MongoClient("mongodb://localhost:27017")
    mongo_collection = mongo_client["llmquoter"][collection_name]

    def _process(sample):
        try:
            if save_to_mongo and model_name:
                doc = mongo_collection.find_one(
                    {id_field: sample[id_field]},
                    {f"scores.{model_name}": 1}
                )
                if doc and doc.get("scores", {}).get(model_name):
                    return {id_field: sample[id_field], "skipped": True}
            chain = get_chain()
            result = evaluate_single(sample["ground_truth"], sample["system_response"], chain, verbose=verbose)
            output = {id_field: sample[id_field]}
            if result:
                output.update({
                    "recall": result["recall"],
                    "precision": result["precision"],
                    "f1": f1_score(result["precision"], result["recall"]),
                })
            else:
                output["error"] = True
            bm25 = bm25_score(sample["ground_truth"], sample["system_response"])
            output["bm25"] = bm25
            fmt = format_score(sample["system_response"])
            output["format_score"] = fmt
            if save_to_mongo and model_name and "error" not in output:
                score_result = {
                    "recall": output.get("recall"),
                    "precision": output.get("precision"),
                    "f1": output.get("f1"),
                    "bm25": output.get("bm25"),
                    "format_score": output.get("format_score", 0.0)
                }
                if mongo_collection.update_one(
                    {id_field: sample[id_field]},
                    {"$set": {f"scores.{model_name}": score_result}}
                ).matched_count > 0:
                    print(f"Saved {sample[id_field][:8]}... [R:{output.get('recall'):.3f} P:{output.get('precision'):.3f} F1:{output.get('f1'):.3f} BM25:{output.get('bm25', 0.0):.3f} FMT:{output.get('format_score', 0.0):.2f}]")
            return output
        except Exception as e:
            print(f"Crash on {sample.get(id_field, '?')[:8]}: {e}")
            traceback.print_exc()
            return {id_field: sample.get(id_field), "error": True}

    if max_workers <= 1:
        results = list(tqdm(map(_process, samples), total=len(samples), desc="Evaluating quotes", unit="sample"))
    else:
        results = list(thread_map(_process, samples, max_workers=max_workers, desc="Evaluating quotes", unit="sample"))
    mongo_client.close()
    return results
