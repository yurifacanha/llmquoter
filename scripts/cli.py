import argparse
from pathlib import Path

from .dataset import (
    fetch_all_from_mongo,
    split_by_field,
    build_dataset_dict,
    push_to_hub,
)
from .mongo_ops import (
    load_existing_dataset,
    get_new_samples_from_mongo,
    convert_doc_to_dict,
    add_samples_to_dataset,
    remove_columns_from_dataset,
    load_raft_quotes_extended,
    add_uuid_to_splits,
    load_and_save_to_mongo,
    update_mongo_from_data_inferences,
)


def upload_hf(repo_name: str, private: bool):
    docs = fetch_all_from_mongo()
    splits = split_by_field(docs)
    dataset_dict = build_dataset_dict(splits)
    push_to_hub(dataset_dict, repo_name, private)
    print(f"Uploaded to https://huggingface.co/datasets/{repo_name}")


def merge_raft(
    source_dataset: str,
    target_dataset: str,
    split: str,
    num_samples: int | None,
    push: bool,
):
    dataset, existing_questions = load_existing_dataset(source_dataset)
    if dataset is None:
        dataset = {}
    new_samples = get_new_samples_from_mongo(existing_questions, split, num_samples)
    if not new_samples:
        print("No new samples to add")
        return
    updated_dataset = add_samples_to_dataset(dataset, new_samples)
    final_dataset = remove_columns_from_dataset(updated_dataset, ["id", "source"])
    if push:
        final_dataset.push_to_hub(target_dataset, private=False, token=True)
        print(f"Pushed to {target_dataset}")


def add_uuids(dataset_name: str, splits: list[str], push: bool):
    add_uuid_to_splits(dataset_name, splits, push)


def save_to_mongo(
    dataset_name: str,
    splits: list[str],
    collection_name: str,
    connection_string: str,
    database_name: str,
):
    count = load_and_save_to_mongo(
        dataset_name, splits, collection_name, connection_string, database_name
    )
    print(f"Saved {count} documents to {collection_name}")


def update_inferences(
    data_dir: str,
    collection_name: str,
    connection_string: str,
    database_name: str,
):
    update_mongo_from_data_inferences(
        data_dir, collection_name, connection_string, database_name
    )
    print("Done updating inferences")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    up = subparsers.add_parser("upload_hf")
    up.add_argument("--repo", default="yurifacanha/hotpotqa_quotes_extended")
    up.add_argument("--private", action="store_true")
    merge = subparsers.add_parser("merge_raft")
    merge.add_argument("--source", default="yurifacanha/RAFTquotes")
    merge.add_argument("--target", default="yurifacanha/RAFTTquotesExtended")
    merge.add_argument("--split", default="train")
    merge.add_argument("--num_samples", type=int, default=None)
    merge.add_argument("--no-push", action="store_true", dest="no_push")
    uuid_p = subparsers.add_parser("add_uuids")
    uuid_p.add_argument("--dataset", default="yurifacanha/RAFTTquotesExtended")
    uuid_p.add_argument("--splits", nargs="+", default=["train", "test"])
    uuid_p.add_argument("--no-push", action="store_true", dest="no_push")
    mongo_p = subparsers.add_parser("save_to_mongo")
    mongo_p.add_argument("--dataset", default="yurifacanha/RAFTTquotesExtended")
    mongo_p.add_argument("--splits", nargs="+", default=["test"])
    mongo_p.add_argument("--collection", default="LLMQuoterTest")
    mongo_p.add_argument("--connection", default="mongodb://localhost:27017")
    mongo_p.add_argument("--database", default="llmquoter")
    inf_p = subparsers.add_parser("update_inferences")
    inf_p.add_argument("--data_dir", default="data")
    inf_p.add_argument("--collection", default="LLMQuoterTest")
    inf_p.add_argument("--connection", default="mongodb://localhost:27017")
    inf_p.add_argument("--database", default="llmquoter")
    args = parser.parse_args()
    if args.command == "upload_hf":
        upload_hf(args.repo, args.private)
    elif args.command == "merge_raft":
        merge_raft(args.source, args.target, args.split, args.num_samples, not args.no_push)
    elif args.command == "add_uuids":
        add_uuids(args.dataset, args.splits, not args.no_push)
    elif args.command == "save_to_mongo":
        save_to_mongo(args.dataset, args.splits, args.collection, args.connection, args.database)
    elif args.command == "update_inferences":
        update_inferences(args.data_dir, args.collection, args.connection, args.database)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
