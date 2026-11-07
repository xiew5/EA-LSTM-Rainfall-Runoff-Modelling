from pathlib import Path
import sys

from neuralhydrology.nh_run import start_run


if __name__ == "__main__":
    config_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("config/neuralhydrology_experment1.yml")
    start_run(config_file=config_file)
