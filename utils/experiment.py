import json
from pathlib import Path
from typing import Callable

import chromadb
import pandas as pd

from chunkers.fixed_token_chunker import FixedTokenChunker

DB_PATH = "data/db"


class Experiment:
    def __init__(
        self,
        corpus_name: str,
    ):
        self.corpus_path: Path = Path(f"data/corpora/{corpus_name}.md")
        self.queries_path: Path = Path("data/queries.csv")

        self.client = chromadb.PersistentClient(path=DB_PATH)

    def evaluate(
        self,
        chunker: FixedTokenChunker,
        embedding_function: Callable,
        retrieved_chunks_num: int,
    ):
        collection = self._create_collection(chunker, embedding_function)
        queries = self._load_queries()

        for query in queries:
            answer = collection.query(
                query_texts=[query["question"]],
                n_results=retrieved_chunks_num,
            )
            print(answer)

        self._delete_collection()

    def _create_collection(
        self, chunker: FixedTokenChunker, embedding_function: Callable
    ):
        with open(self.corpus_path, "r") as f:
            corpus_file = f.read()

        chunks = chunker.split_text(corpus_file)

        collection = self.client.get_or_create_collection(
            self.corpus_path.name, embedding_function=embedding_function
        )

        collection.add(documents=chunks, ids=[str(i) for i in range(len(chunks))])

        return collection

    def _delete_collection(
        self,
    ):
        self.client.delete_collection(self.corpus_path.name)

    def _load_queries(self):
        queries_df = pd.read_csv(self.queries_path)

        queries_df_filtered = queries_df[
            queries_df["corpus_id"] == self.corpus_path.stem
        ]
        queries_df_filtered = queries_df_filtered[["question", "references"]]

        queries = []
        for question, reference_list in queries_df_filtered.values:
            queries.append(
                {"question": question, "references": json.loads(reference_list)}
            )

        return queries
