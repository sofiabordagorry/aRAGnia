from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

TABLE_DIR = Path(__file__).resolve().parents[4] / "data" / "tables"
CHUNK_DIR = Path(__file__).resolve().parents[4] / "data" / "chunks"


def starts_new_row(line: str) -> bool:
    first_column = line.split(",", 1)[0].strip()
    return first_column.isdigit()


def clean_table(input_file: str | Path, output_file: str | Path | None, doc_type: str) -> Path:
    if doc_type not in {"Grupo", "Proyecto"}:
        raise ValueError(f"doc_type debe ser 'Grupo' o 'Proyecto', recibido: {doc_type}")

    if not output_file:
        output_file = input_file
    input_file = Path(input_file)
    output_file = Path(output_file)
    output_file = output_file.with_name(f"{output_file.stem}_table{output_file.suffix}")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with (
        open(input_file, "r", encoding="utf-8", newline="") as infile,
        open(output_file, "w", encoding="utf-8", newline="") as outfile,
    ):

        writer = csv.writer(outfile)
        reader_iter = csv.reader(infile)

        header = next(reader_iter)
        expected_columns = len(header)
        year_idx = header.index("anio")
        form_id_idx = header.index("id_formulario")
        header.insert(0, "row_id")
        header.extend(["tipo_archivo", "id_archivo"])
        writer.writerow(header)

        current_line = ""
        row_counter = 0
        for raw_line in infile:
            line = raw_line.strip()

            if not line:
                continue

            if starts_new_row(line):

                if current_line:
                    row = next(csv.reader([current_line]))

                    if len(row) == expected_columns:
                        year = row[year_idx]
                        form_id = row[form_id_idx]
                        type = "gi" if doc_type == "Grupo" else "proy"
                        doc_id = f"{type}_{year}_{form_id}"

                        row.extend([doc_type, doc_id])
                        row.insert(0, str(row_counter))
                        row_counter += 1
                        writer.writerow(row)

                current_line = line

            else:
                current_line += " " + line

        if current_line:
            row = next(csv.reader([current_line]))

            if len(row) == expected_columns:
                year = row[year_idx]
                form_id = row[form_id_idx]
                doc_id = f"{doc_type}_{year}_{form_id}"
                row.extend([doc_type, doc_id])
                row.insert(0, str(row_counter))
                writer.writerow(row)
    return output_file


def convert_tables_to_chunks() -> None:
    table_dir = Path(TABLE_DIR)

    if not table_dir.exists() or not table_dir.is_dir():
        raise FileNotFoundError(f"No existe la carpeta: {table_dir}")

    csv_files = sorted(table_dir.rglob("*.csv"))

    for csv_path in csv_files:
        csv_path = Path(csv_path)
        build_table_chunks(csv_path)


def build_table_chunks(csv_path: Path) -> None:
    source = csv_path.stem
    chunks: list[dict[str, Any]] = []
    out_dir = Path(CHUNK_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for _, row in enumerate(reader):
            text = " | ".join(f"{key}: {value}" for key, value in row.items() if value)

            metadata = {
                **row,
                "tipo_elemento": "table_row",
                "documento_padre": source,
                "cantidad_tokens": len(text.split()),
            }

            column_id = row.get("row_id")

            chunks.append(
                {
                    "chunk_id": f"{source}#Chunk{column_id}",
                    "text": text,
                    "metadata": metadata,
                }
            )

    payload: dict[str, Any] = {
        "source": source,
        "total_chunks": len(chunks),
        "chunks": chunks,
    }

    out_path = out_dir / f"{source}_chunks.json"

    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
