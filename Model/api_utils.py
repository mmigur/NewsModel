from transformers import pipeline

import pandas as pd


def get_categorired_dataframe(posts: pd.DataFrame, candidate_labels: list[str]) -> dict:
    model = pipeline(task='zero-shot-classification', model='MoritzLaurer/mDeBERTa-v3-base-mnli-xnli')
    model_output = model(list(posts['text']), candidate_labels=candidate_labels)

    categorized = {
        category: {'text': [], 'channel_id': [], 'category': []} for category in candidate_labels
    }

    for index, output_detail in enumerate(model_output):
        predicted_category = output_detail['labels'][0]

        categorized[predicted_category]['text'].append(output_detail['sequence'])
        categorized[predicted_category]['channel_id'].append(posts['channel_id'][index])
        categorized[predicted_category]['category'].append(predicted_category)

    prepared_data = {key: pd.DataFrame.from_dict(value) for key, value in categorized.items()}

    return prepared_data
