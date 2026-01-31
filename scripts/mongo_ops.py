import json
import random
import uuid
from pathlib import Path

from datasets import load_dataset, Dataset, DatasetDict
from pymongo import MongoClient

from mongo import HotPotQAMongo, HotPotQADocument


CONVERSATION_INSTRUCTION = """Instruction: Given the question and the context provide relevant quotes from the context that support the answer. Your answer must be just the quotes, not the entire context.
format: ##begin_quote## quote ##end_quote## for each quote. Do not add anything else other than the quotes.
-------------------------------------------------------
Question: {question}
Context: {context}
Quotes:
"""


def create_conversation(doc: HotPotQADocument) -> list[dict]:
    user_content = CONVERSATION_INSTRUCTION.format(
        question=doc.question,
        context=doc.context
    )
    quotes = doc.quotes or doc.original_quotes or ""
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": quotes}
    ]


def load_existing_dataset(dataset_name: str):
    try:
        dataset = load_dataset(dataset_name)
        all_questions = set()
        for split_data in dataset.values():
            questions = [item.get("question", "").strip().lower() for item in split_data if item.get("question")]
            all_questions.update(questions)
        return dataset, all_questions
    except Exception:
        return None, set()


def get_new_samples_from_mongo(existing_questions: set, split: str, num_samples: int | None):
    with HotPotQAMongo() as mongo:
        all_docs = mongo.find_all(split=split)
    new_samples = []
    for doc in all_docs:
        if not doc.question or not doc.question.strip():
            continue
        question_normalized = doc.question.strip().lower()
        if question_normalized not in existing_questions:
            new_samples.append(doc)
    if num_samples and len(new_samples) > num_samples:
        new_samples = random.sample(new_samples, num_samples)
    return new_samples


def convert_doc_to_dict(doc: HotPotQADocument) -> dict:
    quotes = doc.quotes or doc.original_quotes or ""
    return {
        "conversations": create_conversation(doc),
        "question": doc.question,
        "answer": doc.answer,
        "context": doc.context,
        "quotes": quotes,
    }


def remove_columns_from_dataset(dataset: DatasetDict, columns_to_remove: list[str]):
    updated = {}
    for split_name, split_data in dataset.items():
        columns_to_keep = [c for c in split_data.column_names if c not in columns_to_remove]
        updated[split_name] = split_data.select_columns(columns_to_keep)
    return DatasetDict(updated)


def add_samples_to_dataset(dataset: DatasetDict, new_samples: list):
    new_data = [convert_doc_to_dict(doc) for doc in new_samples]
    dataset_dict = {}
    for split_name in dataset.keys():
        existing_data = []
        for item in dataset[split_name]:
            filtered_item = {k: v for k, v in item.items() if k not in ["id", "source"]}
            existing_data.append(filtered_item)
        dataset_dict[split_name] = Dataset.from_list(existing_data)
    if "train" in dataset_dict:
        existing_train = [item for item in dataset_dict["train"]]
        combined_train = existing_train + new_data
        dataset_dict["train"] = Dataset.from_list(combined_train)
    else:
        dataset_dict["train"] = Dataset.from_list(new_data)
    return DatasetDict(dataset_dict)


def load_raft_quotes_extended(dataset_name: str):
    return load_dataset(dataset_name)


def add_uuid_to_splits(dataset_name: str, splits: list[str], push: bool):
    dataset = load_dataset(dataset_name)
    updated_dataset = {}
    for split_name in dataset.keys():
        if split_name in splits:
            split_data = []
            for item in dataset[split_name]:
                item_dict = dict(item)
                item_dict["uuid"] = str(uuid.uuid4())
                split_data.append(item_dict)
            updated_dataset[split_name] = Dataset.from_list(split_data)
        else:
            existing_data = [dict(item) for item in dataset[split_name]]
            updated_dataset[split_name] = Dataset.from_list(existing_data)
    dataset_dict = DatasetDict(updated_dataset)
    if push:
        dataset_dict.push_to_hub(dataset_name, private=False, token=True)
    return dataset_dict


def load_and_save_to_mongo(
    dataset_name: str,
    splits: list[str],
    collection_name: str,
    connection_string: str,
    database_name: str
):
    dataset = load_dataset(dataset_name)
    documents = []
    for split_name in splits:
        if split_name not in dataset:
            continue
        for item in dataset[split_name]:
            doc = dict(item)
            record = {
                "question": doc.get("question", ""),
                "answer": doc.get("answer", ""),
                "context": doc.get("context", ""),
                "quotes": doc.get("quotes", ""),
                "uuid": doc.get("uuid") or str(uuid.uuid4()),
            }
            documents.append(record)
    client = MongoClient(connection_string)
    collection = client[database_name][collection_name]
    collection.create_index("uuid", unique=True)
    upserted = 0
    for record in documents:
        try:
            collection.replace_one({"uuid": record["uuid"]}, record, upsert=True)
            upserted += 1
        except Exception:
            pass
    client.close()
    return upserted


def update_mongo_from_data_inferences(
    data_dir: str,
    collection_name: str,
    connection_string: str,
    database_name: str
):
    data_path = Path(data_dir)
    json_files = list(data_path.glob("*.json"))
    if not json_files:
        return
    client = MongoClient(connection_string)
    collection = client[database_name][collection_name]
    for json_file in sorted(json_files):
        model_name = json_file.stem
        with open(json_file, "r", encoding="utf-8") as f:
            inferences_data = json.load(f)
        for doc_uuid, quotes in inferences_data.items():
            collection.update_one(
                {"uuid": doc_uuid},
                {"$set": {f"inferences.{model_name}": quotes}}
            )
    client.close()
