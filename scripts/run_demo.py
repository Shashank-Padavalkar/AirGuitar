import argparse
import sys
from pathlib import Path

# Add src to python path for easy running without install
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from airguitarcv.main import run_system

def main():
    project_root = Path(__file__).resolve().parent.parent
    default_config = project_root / "configs" / "default.yaml"
    
    parser = argparse.ArgumentParser(description="Run AirGuitarCV Demo")
    parser.add_argument(
        "--config", type=str, default=str(default_config), help="Path to config file"
    )
    args = parser.parse_args()

    try:
        run_system(args.config)
    except KeyboardInterrupt:
        print("\nDemo stopped by user.")
    except Exception as e:
        print(f"\nError running demo: {e}")

if __name__ == "__main__":
    main()
