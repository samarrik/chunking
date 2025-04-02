import logging
import warnings
from itertools import product
from pathlib import Path

import hydra
from chromadb.utils import embedding_functions
from hydra.core.config_store import ConfigStore
from rich.console import Console
from rich.progress import track

from chunking_experimentation.chunkers import FixedTokenChunker
from chunking_experimentation.config import Config
from chunking_experimentation.experimenter import Experimenter

# Aggressive logger factory reset to ERROR level
original_getLogger = logging.getLogger
def silent_getLogger(*args, **kwargs):
    logger = original_getLogger(*args, **kwargs)
    logger.setLevel(logging.ERROR)
    return logger
logging.getLogger = silent_getLogger

# Ignore all warnings (mostly hydra)
warnings.filterwarnings("ignore")

# Rich console for pretty output
console = Console()

# Hydra configuration
cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: Config) -> None:
    """Orchestrator of the experiments"""
    # Initialize the Experimenter
    experimenter = Experimenter(
        database_path=Path(cfg.database.path),
        corpus_path=Path(cfg.corpus.path),
        queries_path=Path(cfg.queries.path),
        results_data_path=Path(cfg.results.data_path),
        results_figures_path=Path(cfg.results.figures_path),
    )

    # Generate experimental setups from the config
    experimental_setups = list(
        product(
            cfg.experiments.chunk_sizes,
            cfg.experiments.halfway_overlap,
            cfg.experiments.retrieve_count,
        )
    )

    console.print(
        f"[bold green]Running {len(experimental_setups)} experiments[/bold green]"
    )

    # Prepare embedding function
    try:
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=cfg.experiments.embedding_function
        )
    except Exception as e:
        print(
            f"Error loading embedding function: {cfg.experiments.embedding_function} - {e}"
        )
        raise

    # Run the experiments
    for chunk_size, halfway_overlap, retrieve_count in track(
        experimental_setups, description="Conducting experiments..."
    ):
        # Setup the chunker using some of the params
        chunk_overlap = chunk_size // 2 if halfway_overlap else 0
        chunker = FixedTokenChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        experimenter.conduct_experiment(
            chunker=chunker,
            embedding_function=embedding_function,
            retrieve_count=retrieve_count,
        )

    console.print("[bold green]Done![/bold green]")


if __name__ == "__main__":
    main()
