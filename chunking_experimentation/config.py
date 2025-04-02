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


@dataclass
class Experiments:
    embedding_function_name: str
    chunk_sizes: List[int]
    halfway_overlap: List[bool]
    retrieve_count: List[int]


@dataclass
class Config:
    database: Database
    corpus: Corpus
    queries: Queries
    results: Results
    experiments: Experiments
