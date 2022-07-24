"""Package CLI entry point."""
import logging

import defopt

from .datasets import create_rls_genus_dataset, create_rls_species_dataset, create_test_dataset


def run_cli() -> None:
    """Run the command line interface with defopt."""
    logging.basicConfig(
        format="%(asctime)s [%(name)s.%(funcName)s:%(lineno)d] %(levelname)s: %(message)s", level=logging.INFO
    )
    defopt.run([create_rls_genus_dataset, create_rls_species_dataset, create_test_dataset])
