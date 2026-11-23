"""Create a source ZIP with the repository's package structure preserved."""

import argparse
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PATHS = (
    ".streamlit",
    "app.py",
    "ui",
    "src",
    "tests",
    "scripts",
    "benchmarks",
    "evaluation_results",
    "docs",
    "requirements.txt",
    "requirements-ui.txt",
    "README.md",
)
EXCLUDED_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "node_modules",
    "venv",
}
EXCLUDED_FILES = {
    "chroma.sqlite3",
    "chroma.sqlite3-shm",
    "chroma.sqlite3-wal",
    "secrets.toml",
    "credentials.json",
    "service_account.json",
}
EXCLUDED_SUFFIXES = {".key", ".p12", ".pem", ".pfx"}


def _excluded(path: Path, relative_path: Path) -> bool:
    parts = {part.casefold() for part in relative_path.parts}
    name = path.name.casefold()
    if parts.intersection(EXCLUDED_DIRECTORIES):
        return True
    if any("chroma_db" in part for part in parts):
        return True
    if name.startswith(".env"):
        return True
    if name in EXCLUDED_FILES or name.startswith("chroma.sqlite3"):
        return True
    if path.suffix.casefold() in EXCLUDED_SUFFIXES:
        return True
    if path.is_dir() and (path / "chroma.sqlite3").is_file():
        return True
    return False


def create_archive(output_path: Path) -> int:
    ui_package = PROJECT_ROOT / "ui"
    if not (ui_package / "__init__.py").is_file():
        raise FileNotFoundError(f"UI package initializer is missing: {ui_package / '__init__.py'}")

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    archived_files = 0
    with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as archive:
        for package_path in PACKAGE_PATHS:
            source = PROJECT_ROOT / package_path
            if not source.exists():
                continue
            if source.is_file():
                candidates = [source]
            else:
                candidates = []
                for directory, directory_names, file_names in os.walk(source):
                    current_directory = Path(directory)
                    relative_directory = current_directory.relative_to(PROJECT_ROOT)
                    if _excluded(current_directory, relative_directory):
                        directory_names[:] = []
                        continue
                    directory_names[:] = [
                        name
                        for name in directory_names
                        if not _excluded(
                            current_directory / name,
                            (current_directory / name).relative_to(PROJECT_ROOT),
                        )
                    ]
                    candidates.extend(current_directory / name for name in file_names)

            for candidate in candidates:
                relative_path = candidate.relative_to(PROJECT_ROOT)
                if candidate.resolve() == output_path or _excluded(candidate, relative_path):
                    continue
                archive_name = Path(PROJECT_ROOT.name) / relative_path
                archive.write(candidate, archive_name.as_posix())
                archived_files += 1

    return archived_files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / f"{PROJECT_ROOT.name}.zip",
        help="ZIP destination (default: project root)",
    )
    arguments = parser.parse_args()
    archived_files = create_archive(arguments.output)
    print(f"Created {arguments.output.resolve()} with {archived_files} files.")


if __name__ == "__main__":
    main()
