"""PyInstaller entry point — avoids relative import issues."""
from distill.cli import main

if __name__ == "__main__":
    main()
