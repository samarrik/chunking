import logging
import warnings
from pathlib import Path

import hydra
from chromadb.utils import embedding_functions
from hydra.core.config_store import ConfigStore

from chunking_experimentation.chunkers import FixedTokenChunker
from chunking_experimentation.config import Config
from chunking_experimentation.experimenter import Experimenter

# Ignore warnings
warnings.filterwarnings("ignore")

# Aggressive logger factory reset to ERROR level - simpler version
original_getLogger = logging.getLogger


def silent_getLogger(*args, **kwargs):
    logger = original_getLogger(*args, **kwargs)
    logger.setLevel(logging.ERROR)
    return logger


logging.getLogger = silent_getLogger

# Hydra magic to load a config as an object
cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: Config):
    experimenter = Experimenter(
        database_path=Path(cfg.database.path),
        corpus_path=Path(cfg.corpus.path),
        queries_path=Path(cfg.queries.path),
        results_data_path=Path(cfg.results.data_path),
        results_figures_path=Path(cfg.results.figures_path),
    )

    experimental_setups = [
        (size, overlap, num)
        for size in cfg.experiments.chunk_sizes
        for overlap in cfg.experiments.halfway_overlap
        for num in cfg.experiments.retrived_chunks_nums
    ]

    for chunk_size, halfway_overlap, retrived_chunks_num in experimental_setups:
        # Prepare the chunker
        chunk_overlap = chunk_size // 2 if halfway_overlap else 0
        chunker = FixedTokenChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Prepare the embedding function
        try:
            embedding_function = (
                embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=cfg.experiments.embedding_function
                )
            )
        except Exception as e:
            print(f"Error loading embedding function: {e}")
            raise e

        # Conduct the experiment
        experimenter.conduct_experiment(
            chunker=chunker,
            embedding_function=embedding_function,
            retrived_chunks_num=retrived_chunks_num,
        )


if __name__ == "__main__":
    main()
