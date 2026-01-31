INFERENCE_PROMPT = """Given the question and the context provide relevant quotes from the context that support the answer.
Your answer must be just the quotes, not the entire context.
Format: ##begin_quote## quote ##end_quote## for each quote.
Do not add anything else other than the quotes.

Question: {question}
Context: {context}

Quotes:
"""

EVALUATOR_PROMPT = """You are an expert evaluator that compares two sets of quotes and calculates semantic recall and precision.

**Ground Truth Quotes:**
{ground_truth}

**System Response Quotes:**
{system_response}

Use the SAME matching rule for both metrics: two quotes match if they convey the same main information, even if wording differs.
Matching is symmetric: if quote A matches quote B, then B matches A.

- Recall: Fraction (0.0 to 1.0) of ground truth quotes that have at least one matching quote in the system response
- Precision: Fraction (0.0 to 1.0) of system response quotes that have at least one matching quote in the ground truth

IMPORTANT: If recall > 0, then precision must also be > 0 (a matched ground truth quote implies a matching system quote).
If the system response is empty or has no valid quotes, precision = 0 and recall = 0 (if GT has quotes) or 1 (if GT is empty).
If ground truth is empty, recall = 1 and precision = 0.

"""