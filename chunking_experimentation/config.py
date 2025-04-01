from dataclasses import dataclass
from typing import List


@dataclass
class Database:
    path: str


@dataclass
class Corpus:
    path: str


@dataclass
class Queries:
    path: str


@dataclass
class Results:
    data_path: str
    figures_path: str


@dataclass
class Experiments:
    embedding_function: str
    chunk_sizes: List[int]
    halfway_overlap: List[bool]
    retrived_chunks_nums: List[int]


@dataclass
class Config:
    database: Database
    corpus: Corpus
    queries: Queries
    results: Results
    experiments: Experiments
