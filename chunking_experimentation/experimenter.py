import json
from pathlib import Path

import chromadb
import pandas as pd
from chromadb.utils import embedding_functions

from chunking_experimentation.chunkers import FixedTokenChunker


class Experimenter:
    def __init__(
        self,
        *,
        database_path: Path,
        corpus_path: Path,
        queries_path: Path,
        results_data_path: Path,
        results_figures_path: Path,
    ):
        # Load the paths
        self.corpus_path: Path = corpus_path
        self.queries_path: Path = queries_path
        self.results_data_path: Path = results_data_path
        self.results_figures_path: Path = results_figures_path

        # Setup the db client
        self.client = chromadb.PersistentClient(path=str(database_path))

    def conduct_experiment(
        self,
        chunker: FixedTokenChunker,
        embedding_function: embedding_functions.SentenceTransformerEmbeddingFunction,
        retrived_chunks_num: int,
    ):
        collection = self._create_collection(
            chunker=chunker, embedding_function=embedding_function
        )
        queries = self._load_queries()

        for query in queries:
            answer = collection.query(
                query_texts=[query["question"]],
                n_results=retrived_chunks_num,
            )
        print(f"All {len(queries)} queries have been executed")

    def _create_collection(
        self,
        chunker: FixedTokenChunker,
        embedding_function: embedding_functions.SentenceTransformerEmbeddingFunction,
    ):
        with open(self.corpus_path, "r") as f:
            corpus_file = f.read()

        chunks = chunker.split_text(corpus_file)

        collection = self.client.get_or_create_collection(
            self.corpus_path.name, embedding_function=embedding_function
        )

        collection.upsert(documents=chunks, ids=[str(i) for i in range(len(chunks))])

        return collection

    def _load_queries(self):
        df = pd.read_csv(self.queries_path)
        filtered = df.loc[
            df["corpus_id"] == self.corpus_path.stem, ["question", "references"]
        ]

        queries = []
        for _, row in filtered.iterrows():
            queries.append(
                {
                    "question": row["question"],
                    "references": json.loads(row["references"]),
                }
            )

        return queries
