import evaluate
import numpy as np


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # predictions, labels = eval_pred
    # Assuming predictions and labels are lists of strings

    # Load metrics
    perplexity = evaluate.load('perplexity')
    rouge = evaluate.load('rouge')
    bertscore = evaluate.load('bertscore')

    # Compute metrics
    perplexity_result = perplexity.compute(predictions=predictions, references=labels)
    rouge_result = rouge.compute(predictions=predictions, references=labels)
    bertscore_result = bertscore.compute(predictions=predictions, references=labels)

    return {
        'perplexity': perplexity_result['perplexity'],
        'rouge': rouge_result,
        'bertscore': bertscore_result
    }