from __future__ import annotations

import logging
import sys

from src.feature_pipeline_core import FeatureEngineeringPipeline


def main() -> None:
    print("eorking")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,  
    )
    logging.info("Starting feature engineering pipeline...")
    FeatureEngineeringPipeline().run_all()
    logging.info("Feature engineering pipeline completed.")


if __name__ == "__main__":
    main()
