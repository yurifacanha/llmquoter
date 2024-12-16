import dspy

class SemanticRecallPrecision(dspy.Signature):
    """
    Signature class for calculating semantic recall and precision.
    
    Attributes:
        ground_truth (str): The ground truth quotes.
        system_response (str): The quotes provided by the system's response.
        recall (float): Fraction (out of 1.0) of how many quotes from the ground truth 
                        are present in the system response.
        precision (float): Fraction (out of 1.0) of how many quotes from the system response 
                           are present in the ground truth.
    """
    ground_truth: str = dspy.InputField()
    system_response: str = dspy.InputField()
    recall: float = dspy.OutputField(desc="Fraction (out of 1.0) of how many quotes from the ground truth are present in the system response")
    precision: float = dspy.OutputField(desc="Fraction (out of 1.0) of how many quotes from the system response are present in the ground truth")


def f1_score(precision, recall):
    """
    Compute the F1 score from precision and recall values.

    Args:
        precision (float): Precision value.
        recall (float): Recall value.

    Returns:
        float: The F1 score, calculated as 2 * (precision * recall) / (precision + recall).
    """
    precision, recall = max(0.0, min(1.0, precision)), max(0.0, min(1.0, recall))
    return 0.0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall)


class RecallPrecisionMetric(dspy.Module):
    """
    Module for calculating recall and precision metrics using semantic comparison.
    """
    def __init__(self):
        self.module = dspy.ChainOfThought(SemanticRecallPrecision)

    def forward(self, ground_truth, system_response):
        """
        Forward method to compute recall and precision scores.

        Args:
            ground_truth (str): Ground truth quotes.
            system_response (str): System-generated quotes.

        Returns:
            dict: Recall and precision scores.
        """
        scores = self.module(
            ground_truth=ground_truth,
            system_response=system_response
        )
        return scores


class SemanticAccuracy(dspy.Signature):
    """
    Signature class for calculating semantic accuracy.
    
    Attributes:
        question (str): The input question.
        answer (str): The system-generated answer.
        expected_answer (str): The ground truth answer.
        accuracy (float): Fraction (out of 1.0) representing how close the system's 
                          answer is to the expected answer.
    """
    question: str = dspy.InputField()
    answer: str = dspy.InputField()
    expected_answer: str = dspy.InputField()
    accuracy = dspy.OutputField(desc="Fraction (out of 1.0) representing how close the system's answer is to the expected answer")


class AccuracyMetric(dspy.Module):
    """
    Module for calculating accuracy metrics.
    """
    def __init__(self):
        self.module = dspy.ChainOfThought(SemanticAccuracy)

    def forward(self, question, answer, expected_answer):
        """
        Forward method to compute accuracy score.

        Args:
            question (str): The input question.
            answer (str): The system-generated answer.
            expected_answer (str): The ground truth answer.

        Returns:
            dict: Accuracy score.
        """
        scores = self.module(
            question=question,
            answer=answer,
            expected_answer=expected_answer
        )
        return scores


def evaluate_quotes(samples):
    """
    Evaluate recall and precision metrics for a list of samples.

    Args:
        samples (list[dict]): List of samples containing ground truth and system responses.

    Returns:
        list[dict]: List of results with recall, precision, and F1 scores for each sample.
    """
    metric = SemanticRecallPrecision()
    results = []
    for sample in samples:
        r = {}
        original = metric(sample['quotes'], sample['result_original'])
        llmquoter = metric(sample['ground_truth'], sample['system']['result_llmquoter'])
        r['id'] = sample['id']
        r['metric_original'] = {
            'recall': original.recall,
            'precision': original.precision,
            'f1': f1_score(original.recall, original.precision)
        }
        r['metric_raft'] = {
            'recall': llmquoter.recall,
            'precision': llmquoter.precision,
            'f1': f1_score(llmquoter.recall, llmquoter.precision)
        }
        results.append(r)
    return results


def evaluate_answers(samples):
    """
    Evaluate accuracy metrics for a list of samples.

    Args:
        samples (list[dict]): List of samples containing questions, answers, and expected answers.

    Returns:
        list[dict]: List of results with accuracy scores for each sample.
    """
    metric = AccuracyMetric()
    results = []
    for sample in samples:
        llama1b_context = metric(sample['question'], sample['llama1b_answers']['context'], sample['answer'])
        llama1b_quotes = metric(sample['question'], sample['llama1b_answers']['quotes'], sample['answer'])
        llama3b_context = metric(sample['question'], sample['llama3b_answers']['context'], sample['answer'])
        llama3b_quotes = metric(sample['question'], sample['llama3b_answers']['quotes'], sample['answer'])
        gpt35_context = metric(sample['question'], sample['gpt_answers']['context'], sample['answer'])
        gpt35_quotes = metric(sample['question'], sample['gpt_answers']['quotes'], sample['answer'])

        accuracy = {
            'id': sample['id'],
            'accuracy_llama1b': {
                'context': llama1b_context.accuracy,
                'quotes': llama1b_quotes.accuracy
            },
            'accuracy_llama3b': {
                'context': llama3b_context.accuracy,
                'quotes': llama3b_quotes.accuracy
            },
            'accuracy_gpt': {
                'context': gpt35_context.accuracy,
                'quotes': gpt35_quotes.accuracy
            }
        }
        results.append(accuracy)
    return results
