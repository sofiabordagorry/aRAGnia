from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

from institutional_graphrag.ingest.file_namer import generate_new_filename
from institutional_graphrag.ingest.type_converter import odt_bytes_to_pdf


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS_DIR = BASE_DIR / "data" / "corpus"


def is_valid_project_path(rel_path: str) -> bool:
    parts = rel_path.replace("\\", "/").split("/")
    root = parts[0] if parts else ""
    is_proyecto = re.match(r"^proyectos[_\s]?\d{4}", root, flags=re.IGNORECASE) is not None
    is_grupo = (
        re.search(r"grupos", root, flags=re.IGNORECASE) is not None
        or re.match(r"^gi[_\s]", root, flags=re.IGNORECASE) is not None
    )

    if not is_proyecto and not is_grupo:
        return False

    return any(part.isdigit() for part in parts[1:-1])


def process_projects(input_dir: Path, output_dir: Path) -> tuple[int, int]:
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de entrada: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    all_files = [p for p in input_dir.rglob("*") if p.is_file()]
    valid_files: list[tuple[str, bytes]] = []
    invalid_roots: set[str] = set()

    for file_path in all_files:
        # Si el input es directamente "proyectos_2014", el frontend valida
        # paths comenzando por esa raíz. Por eso la reincorporamos acá.
        rel_inside = file_path.relative_to(input_dir).as_posix()
        rel_path = f"{input_dir.name}/{rel_inside}"

        if file_path.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(file_path) as zf:
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        inner_path = info.filename.replace("\\", "/")
                        if inner_path.startswith("__MACOSX/"):
                            continue
                        if any(part.startswith(".") for part in inner_path.split("/")):
                            continue

                        if is_valid_project_path(inner_path):
                            content = zf.read(info)
                            valid_files.append((inner_path, content))
                        else:
                            parts = inner_path.split("/")
                            if parts:
                                invalid_roots.add(parts[0])
            except zipfile.BadZipFile:
                invalid_roots.add(file_path.name)
        elif is_valid_project_path(rel_path):
            content = file_path.read_bytes()
            valid_files.append((rel_path, content))
        else:
            parts = rel_path.split("/")
            if parts:
                invalid_roots.add(parts[0])

    if all_files and not valid_files:
        roots = ", ".join(sorted(invalid_roots))
        raise ValueError(
            "No se encontraron archivos con estructura válida de proyectos/grupos. "
            f"Raíces detectadas: {roots}"
        )

    processed = 0
    skipped = len(all_files) - len(valid_files)

    for rel_path, content in valid_files:
        new_filename = generate_new_filename(rel_path)
        final_content = content

        if Path(new_filename).suffix.lower() == ".odt":
            final_content = odt_bytes_to_pdf(content)
            new_filename = str(Path(new_filename).with_suffix(".pdf"))

        target_path = output_dir / new_filename
        target_path.write_bytes(final_content)
        processed += 1

    return processed, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Valida estructura de carpetas de proyectos/grupos, renombra archivos "
            "y los guarda en data/corpus."
        )
    )
    parser.add_argument(
        "input_path",
        help=(
            "Ruta a la carpeta de entrada que contiene archivos de proyectos/grupos "
            "con estructura de carga del frontend."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default=str(DEFAULT_CORPUS_DIR),
        help="Ruta de salida para guardar archivos renombrados (por defecto: data/corpus).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_path)
    output_dir = Path(args.output)

    processed, skipped = process_projects(input_dir, output_dir)
    print("=" * 60)
    print("Proceso finalizado")
    print(f"Archivos procesados: {processed}")
    print(f"Archivos ignorados por estructura inválida: {skipped}")
    print(f"Salida: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
