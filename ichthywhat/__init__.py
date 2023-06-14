"""Package CLI entry point."""


def run_cli() -> None:
    """Run the command line interface with defopt."""
    import logging

    import defopt

    from ichthywhat.datasets import (
        create_rls_genus_dataset,
        create_rls_species_dataset,
        create_test_dataset,
    )
    from ichthywhat.training import export_learner_to_onnx, train_app_model

    logging.basicConfig(
        format="%(asctime)s [%(name)s.%(funcName)s:%(lineno)d] %(levelname)s: "
        "%(message)s",
        level=logging.INFO,
    )
    defopt.run(
        [
            create_rls_genus_dataset,
            create_rls_species_dataset,
            create_test_dataset,
            export_learner_to_onnx,
            train_app_model,
        ]
    )
