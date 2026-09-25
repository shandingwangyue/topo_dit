import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from topo_dit.train_cli import main

if __name__ == "__main__":
    main(phase=2)
