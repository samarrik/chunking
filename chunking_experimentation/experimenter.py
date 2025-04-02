import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import chromadb
import numpy as np
import pandas as pd
from chromadb.api.models.Collection import Collection
from chromadb.utils import embedding_functions

from chunking_experimentation.chunkers import FixedTokenChunker
from chunking_experimentation.ranges import (
    intersect_ranges,
    sum_of_ranges,
    union_ranges,
)
from chunking_experimentation.utils import rigorous_document_search


@dataclass
class ExperimentResult:
    """Structured container for storing experiment results and parameters."""
    corpus: str
    embedding_function: str
    chunk_size: int
    chunk_overlap: int
    retrieve_count: int
    iou: float
    precision: float
    recall: float


class Experimenter:
    def __init__(
        self,
        *,
        database_path: Path,
        corpus_path: Path,
        queries_path: Path,
        results_data_path: Path,
        embedding_function_name: str
    ):
        self.corpus_path = corpus_path
        self.queries_path = queries_path
        self.results_data_path = results_data_path
        self.embedding_function_name = embedding_function_name

        self.client = chromadb.PersistentClient(path=str(database_path))

    def conduct_experiment(
        self,
        chunker: FixedTokenChunker,
        embedding_function: embedding_functions.SentenceTransformerEmbeddingFunction,
        retrieve_count: int,
    ) -> ExperimentResult:
        """Run a single chunking experiment with given parameters."""
        chunk_collection, queries_collection = self._compose_collections(
            chunker, embedding_function
        )

        iou, precision, recall = self._compute_metrics(
            chunk_collection, queries_collection, retrieve_count
        )

        result = ExperimentResult(
            corpus=self.corpus_path.stem,
            embedding_function=self.embedding_function_name,
            chunk_size=chunker._chunk_size,
            chunk_overlap=chunker._chunk_overlap,
            retrieve_count=retrieve_count,
            iou=float(iou),
            precision=float(precision),
            recall=float(recall),
        )

        self._save_result(result)
        return result

    def _save_result(self, result: ExperimentResult) -> None:
        """Append experiment result to CSV file, creating it if doesn't exist."""
        self.results_data_path.mkdir(parents=True, exist_ok=True)
        output_file = self.results_data_path / "results.csv"

        result_row = {
            "corpus": result.corpus,
            "embedding_function": result.embedding_function,
            "chunk_size": result.chunk_size,
            "chunk_overlap": result.chunk_overlap,
            "retrieve_count": result.retrieve_count,
            "iou": result.iou,
            "precision": result.precision,
            "recall": result.recall,
        }

        df_new = pd.DataFrame([result_row])

        # If file exists, append; if not, create new
        if output_file.exists():
            df_new.to_csv(output_file, mode='a', header=False, index=False)
        else:
            df_new.to_csv(output_file, mode='w', header=True, index=False)

    def _compose_collections(
        self,
        chunker: FixedTokenChunker,
        embedding_function: embedding_functions.SentenceTransformerEmbeddingFunction,
    ) -> Tuple[Collection, Collection]:
        """Create or retrieve collections for chunks and queries, populating if empty."""
        # Unique collection name includes all parameters to avoid mixing data
        chunk_collection = self.client.get_or_create_collection(
            name=f"{self.corpus_path.stem}_{self.embedding_function_name}_{chunker._chunk_size}_{chunker._chunk_overlap}",
            embedding_function=embedding_function,
        )
        if len(chunk_collection.get()["ids"]) == 0:
            chunks, metas_c = self._chunk_corpus(self.corpus_path, chunker)
            ids = [str(i) for i in range(len(chunks))]
            chunk_collection.upsert(documents=chunks, metadatas=metas_c, ids=ids)

        queries_collection = self.client.get_or_create_collection(
            name=f"{self.queries_path.stem}_{self.embedding_function_name}",
            embedding_function=embedding_function
        )
        if len(queries_collection.get()["ids"]) == 0:
            queries, metas_q = self._load_queries()
            ids = [str(i) for i in range(len(queries))]
            queries_collection.upsert(documents=queries, metadatas=metas_q, ids=ids)

        return chunk_collection, queries_collection

    def _chunk_corpus(
        self, corpus_path: Path, chunker: FixedTokenChunker
    ) -> Tuple[List[str], List[Dict[str, int]]]:
        """Split corpus into chunks and generate position metadata for each chunk."""
        with open(corpus_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        chunks = chunker.split_text(text)
        metas = []
        for chunk in chunks:
            _, start, end = rigorous_document_search(text, chunk)
            metas.append({"start_index": start, "end_index": end})

        return chunks, metas

    def _load_queries(self) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Load queries and their reference ranges from CSV file."""
        df = pd.read_csv(self.queries_path)
        filtered = df.loc[
            df["corpus_id"] == self.corpus_path.stem, ["question", "references"]
        ]

        queries, metas = [], []
        for _, row in filtered.iterrows():
            queries.append(row["question"])
            metas.append({"references": row["references"]})

        return queries, metas

    def _compute_metrics(
        self,
        chunk_collection: Collection,
        queries_collection: Collection,
        retrieve_count: int,
    ) -> Tuple[float, float, float]:
        """Calculate IoU, precision, and recall metrics for retrieved chunks."""
        queries_data = queries_collection.get(
            include=["documents", "embeddings", "metadatas"]
        )
        retrievals = chunk_collection.query(
            query_embeddings=queries_data["embeddings"], 
            n_results=retrieve_count
        )

        all_metrics = []
        for idx in range(len(queries_data["embeddings"])):
            # Parse reference ranges from query metadata
            references = json.loads(queries_data["metadatas"][idx]["references"])
            ref_ranges = [
                (int(ref["start_index"]), int(ref["end_index"])) for ref in references
            ]
            ref_ranges = union_ranges(ref_ranges)

            retrieved_meta = retrievals["metadatas"][idx]
            if not retrieved_meta:
                all_metrics.append((0, 0, 0))
                continue

            # Get ranges of retrieved chunks and merge overlapping ones
            chunk_ranges = [
                (meta["start_index"], meta["end_index"]) for meta in retrieved_meta
            ]
            chunk_ranges = union_ranges(chunk_ranges)

            # Find all intersections between reference and retrieved ranges
            intersections = [
                inter
                for cr in chunk_ranges
                for rr in ref_ranges
                if (inter := intersect_ranges(cr, rr))
            ]

            if not intersections:
                all_metrics.append((0, 0, 0))
                continue

            # Calculate overlap metrics
            inter_union = union_ranges(intersections)
            inter_length = sum_of_ranges(inter_union)
            retrieved_length = sum_of_ranges(chunk_ranges)
            ref_length = sum_of_ranges(ref_ranges)
            union_length = retrieved_length + ref_length - inter_length

            precision = inter_length / retrieved_length if retrieved_length > 0 else 0
            recall = inter_length / ref_length if ref_length > 0 else 0
            iou = inter_length / union_length if union_length > 0 else 0

            all_metrics.append((iou, precision, recall))

        if not all_metrics:
            return 0, 0, 0

        iou_values, precision_values, recall_values = zip(*all_metrics)
        return (np.mean(iou_values), np.mean(precision_values), np.mean(recall_values))
