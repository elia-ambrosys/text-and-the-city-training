import pandas as pd
import os
import mlflow
import mlflow.transformers
import functools
import transformers
import datasets
import numpy as np
import sklearn.model_selection
import torch
import argparse


def _get_required_env_var(name: str):
    value = os.getenv(name)
    if value is None:
        raise NameError(f"Environment variable '{name}' not set")
    return value


DATA_FILE = _get_required_env_var("DATA_FILE")


def _clean_data(data_frame: pd.DataFrame):
    data_frame = data_frame[data_frame['Name des pdf Dokuments'].notnull()]
    no_duplicated_df = data_frame.drop_duplicates(subset=['Name des pdf Dokuments'])[
        ['Name des pdf Dokuments', "Inhalt"]]

    def create_labels_column(row):
        return [1.0 if len(
            data_frame[(data_frame['Name des pdf Dokuments'] == row['Name des pdf Dokuments']) & (
                        data_frame["Ziel"] == label)].values) else 0.0
                for label in labels]

    no_duplicated_df['labels'] = no_duplicated_df.apply(create_labels_column, axis=1)
    no_duplicated_df = no_duplicated_df.rename(columns={'Inhalt': 'text'})[["text", "labels"]]
    data_frame = no_duplicated_df.reset_index(drop=True)
    return data_frame


def tok_func(x, tokenizer):
    bert_max_token_n = 512
    return tokenizer(x["text"], padding="max_length", truncation=True, max_length=bert_max_token_n)


def get_dataset(ds, tok_func, tokenizer, indices_train, indices_test, train=True):
    datasets_ds = datasets.Dataset.from_pandas(df)
    tok_function_partial = functools.partial(tok_func, tokenizer=tokenizer)
    tok_ds = datasets_ds.map(tok_function_partial, batched=True)
    if train:
        return datasets.DatasetDict(
            {
                "train": tok_ds.select(indices_train),
                "test": tok_ds.select(indices_test),
            }
        )
    else:
        return tok_ds


def multi_label_metrics(predictions, labels, threshold=0.5):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    y_true = labels
    f1_micro_average = sklearn.metrics.f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = sklearn.metrics.roc_auc_score(y_true, y_pred, average='micro')
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    mlflow.log_metrics(metrics)
    return metrics


def compute_metrics(p: transformers.EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions,
                                           tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds,
        labels=p.label_ids)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", "-b", type=int, default=128)
    parser.add_argument("--epochs", "-e", type=int, default=4)

    args = parser.parse_args()

    batch_size = args.batch_size
    epochs = args.epochs

    df = pd.read_excel(DATA_FILE, engine='openpyxl')
    labels = [label for label in df['Ziel'].unique()]
    id2label = {idx: label for idx, label in enumerate(labels)}
    label2id = {label: idx for idx, label in enumerate(labels)}
    df = _clean_data(data_frame=df)
    indices_train, indices_test = sklearn.model_selection.train_test_split(
        np.arange(df.shape[0]),
        random_state=42,
        shuffle=True,
        test_size=0.2,
    )
    HUGGING_FACE_NAME = "bert-base-multilingual-cased"
    tokenizer = transformers.AutoTokenizer.from_pretrained(HUGGING_FACE_NAME)
    dataset = get_dataset(df, tok_func, tokenizer, indices_train, indices_test)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(HUGGING_FACE_NAME,
                                                                            problem_type="multi_label_classification",
                                                                            num_labels=len(labels),
                                                                            id2label=id2label,
                                                                            label2id=label2id)
    parameters = {
        "learning_rate": 8e-5,
        "batch_size": batch_size,
        "weight_decay": 0.01,
        "epochs": epochs,
    }
    args = transformers.TrainingArguments(
        "parameters",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=parameters["learning_rate"],
        per_device_train_batch_size=parameters["batch_size"],
        per_device_eval_batch_size=parameters["batch_size"],
        num_train_epochs=parameters["epochs"],
        weight_decay=parameters["weight_decay"],
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )
    trainer = transformers.Trainer(
        model,
        args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    with mlflow.start_run():
        mlflow.log_params(parameters)
        trainer.train()
        components = {
            "model": model,
            "tokenizer": tokenizer,
        }
        mlflow.transformers.log_model(
            transformers_model=components,
            artifact_path=f"{HUGGING_FACE_NAME}-text-and-the-city",
        )
