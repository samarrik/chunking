import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import chromadb
import numpy as np
import pandas as pd
import yaml
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

        # Clear the results if present
        if self.results_data_path.exists:
            shutil.rmtree(self.results_data_path)
        self.results_data_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(database_path))

    def conduct_experiment(
        self,
        chunker: FixedTokenChunker,
        embedding_function: embedding_functions.SentenceTransformerEmbeddingFunction,
        retrieve_count: int,
    ) -> ExperimentResult:
        """Conduct a single experiment with the given parameters and return results."""
        chunk_collection, queries_collection = self._compose_collections(
            chunker, embedding_function
        )

        iou, precision, recall = self._compute_metrics(
            chunk_collection, queries_collection, retrieve_count
        )

        result = ExperimentResult(
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
        """Save experiment result to YAML file."""
        self.results_data_path.mkdir(parents=True, exist_ok=True)
        output_file = self.results_data_path / "output.yaml"

        result_dict = {
            "chunk_size": result.chunk_size,
            "chunk_overlap": result.chunk_overlap,
            "retrieve_count": result.retrieve_count,
            "metrics": {
                "iou": result.iou,
                "precision": result.precision,
                "recall": result.recall,
            },
        }

        # Load existing results and append the fresh ones
        results = []
        if output_file.exists():
            with open(output_file, "r") as f:
                results = yaml.safe_load(f) or []
        results.append(result_dict)

        # Write results with improved formatting
        with open(output_file, "w") as f:
            yaml.dump(
                results,
                f,
                sort_keys=False,  # keeps the order of the els the same
                indent=2,
            )

    def _compose_collections(
        self,
        chunker: FixedTokenChunker,
        embedding_function: embedding_functions.SentenceTransformerEmbeddingFunction,
    ) -> Tuple[Collection, Collection]:
        """Create and populate collections for chunks and queries."""
        # Create & populate the collection of chunks
        chunk_collection = self.client.get_or_create_collection(
            name=f"{self.corpus_path.stem}_{self.embedding_function_name}_{chunker._chunk_size}_{chunker._chunk_overlap}",
            embedding_function=embedding_function,
        )
        if len(chunk_collection.get()["ids"]) == 0:
            chunks, metas_c = self._chunk_corpus(self.corpus_path, chunker)
            ids = [str(i) for i in range(len(chunks))]
            chunk_collection.upsert(documents=chunks, metadatas=metas_c, ids=ids)

        # Create & populate the collection of queries
        queries_collection_name = self.queries_path.stem
        queries_collection = self.client.get_or_create_collection(
            name=f"{queries_collection_name}_{self.embedding_function_name}", embedding_function=embedding_function
        )
        if len(queries_collection.get()["ids"]) == 0:
            queries, metas_q = self._load_queries()
            ids = [str(i) for i in range(len(queries))]
            queries_collection.upsert(documents=queries, metadatas=metas_q, ids=ids)

        return chunk_collection, queries_collection

    def _chunk_corpus(
        self, corpus_path: Path, chunker: FixedTokenChunker
    ) -> Tuple[List[str], List[Dict[str, int]]]:
        """Chunk a corpus and generate metadata."""
        with open(corpus_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Conduct chinking and get metadata
        chunks = chunker.split_text(text)
        metas = []
        for chunk in chunks:
            _, start, end = rigorous_document_search(text, chunk)
            metas.append({"start_index": start, "end_index": end})

        return chunks, metas

    def _load_queries(self) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Load and parse queries from CSV file."""
        df = pd.read_csv(self.queries_path)
        filtered = df.loc[
            df["corpus_id"] == self.corpus_path.stem, ["question", "references"]
        ]

        # Get queries and metadata - (unparsed references)
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
        """Compute evaluation metrics for the given collections."""
        queries_data = queries_collection.get(
            include=["documents", "embeddings", "metadatas"]
        )
        retrievals = chunk_collection.query(
            query_embeddings=queries_data["embeddings"], n_results=retrieve_count
        )

        # Process each queries
        all_metrics = []
        for idx in range(len(queries_data["embeddings"])):
            # Get reference ranges
            references = json.loads(queries_data["metadatas"][idx]["references"])
            ref_ranges = [
                (int(ref["start_index"]), int(ref["end_index"])) for ref in references
            ]
            ref_ranges = union_ranges(ref_ranges)

            # Get retrieved chunk ranges
            retrieved_meta = retrievals["metadatas"][idx]
            if not retrieved_meta:
                all_metrics.append((0, 0, 0))  # iou, precision, recall
                continue
            chunk_ranges = [
                (meta["start_index"], meta["end_index"]) for meta in retrieved_meta
            ]
            chunk_ranges = union_ranges(chunk_ranges)

            # Find intersections
            intersections = [
                inter
                for cr in chunk_ranges
                for rr in ref_ranges
                if (inter := intersect_ranges(cr, rr))
            ]

            if not intersections:
                all_metrics.append((0, 0, 0))
                continue

            inter_union = union_ranges(intersections)
            inter_length = sum_of_ranges(inter_union)

            retrieved_length = sum_of_ranges(chunk_ranges)
            ref_length = sum_of_ranges(ref_ranges)
            union_length = retrieved_length + ref_length - inter_length

            # Calculate metrics
            precision = inter_length / retrieved_length if retrieved_length > 0 else 0
            recall = inter_length / ref_length if ref_length > 0 else 0
            iou = inter_length / union_length if union_length > 0 else 0

            all_metrics.append((iou, precision, recall))

        if not all_metrics:
            return 0, 0, 0

        # Reform and get means
        iou_values, precision_values, recall_values = zip(*all_metrics)
        return (np.mean(iou_values), np.mean(precision_values), np.mean(recall_values))
