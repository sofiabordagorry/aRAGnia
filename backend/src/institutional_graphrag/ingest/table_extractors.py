import re
from pathlib import Path

import pandas as pd
import tabula
from odf import teletype
from odf.opendocument import load
from odf.table import Table, TableRow
from odf.text import P

CUSTOM_HEADER_ODT = [
    "Id",
    "Fragmento de título",
    "Apellido 1",
    "Apellido 2",
    "Area",
    "Propuesta",
    "Fid inf.",
    "Apellido 1 Inf",
    "Apellido 2 Inf",
    "Informe",
]

CUSTOM_HEADER_PDF = [
    "Id",
    "Título",
    "Diciplina",
    "Nombre Responsable",
    "Apellido Responsable",
    "Grado",
    "Servicio",
    "Nombre Responsable_2",
    "Apellido Responsable_2",
    "Grado_2",
    "Servicio_2",
    "Monto",
]


# ---------- limpieza ----------
def _clean_cell(x: str) -> str:
    s = (x or "").replace("_x000D_", " ")
    s = s.replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", s).strip()


def clean_x000d_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_clean_cell(str(c)) for c in df.columns]
    for c in df.columns:
        df[c] = df[c].map(lambda v: v if pd.isna(v) else _clean_cell(str(v)))
    return df


# ---------- headers ----------
def make_unique_columns(cols: list[str]) -> list[str]:
    seen, out = {}, []
    for c in cols:
        c = c or ""
        k = seen.get(c, 0)
        out.append(c if k == 0 else f"{c}_{k}")
        seen[c] = k + 1
    return out


def is_header_row(row: list[str]) -> bool:
    return bool(row and row[0] and _clean_cell(row[0]).lower().startswith("id"))


def normalize_header(raw: list[str]) -> list[str]:
    header = [_clean_cell(h) for h in raw]
    header = [h if h else f"col_{i}" for i, h in enumerate(header)]
    return make_unique_columns(header)


def find_header_idx(rows: list[list[str]]) -> int | None:
    for i, r in enumerate(rows):
        if is_header_row(r):
            return i
    return None


def build_df_from_rows(data_rows: list[list[str]], header: list[str]) -> pd.DataFrame:
    n = len(header)
    fixed = [(r[:n] + [""] * (n - len(r))) for r in data_rows]
    return (
        pd.DataFrame(fixed, columns=header)
        .replace("", pd.NA)
        .dropna(how="all")
        .reset_index(drop=True)
    )


# ---------- odf / ods ----------
def extract_row_cells(row) -> list[str]:
    cells: list[str] = []
    for node in row.childNodes:
        tag = getattr(node, "tagName", None)
        if tag not in ("table:table-cell", "table:covered-table-cell"):
            continue

        repeat = node.getAttribute("numbercolumnsrepeated")
        repeat = int(repeat) if repeat else 1

        if tag == "table:covered-table-cell":
            text = ""
        else:
            parts = [teletype.extractText(p) or "" for p in node.getElementsByType(P)]
            text = _clean_cell(" ".join(parts))

        cells.extend([text] * repeat)

    while cells and cells[-1] == "":
        cells.pop()
    return cells


def _is_auto_col(name: str) -> bool:
    return re.match(r"^col_\d+$", str(name)) is not None


def fix_merged_header_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Mantengo la misma lógica que tenías (solo afecta "Primer/Segundo Responsable")
    df = df.copy()
    cols = list(df.columns)

    i = 1
    while i < len(cols):
        prev, curr = cols[i - 1], cols[i]
        if _is_auto_col(curr) and prev in {"Primer Responsable", "Segundo Responsable"}:
            left = df[prev].fillna("").astype(str)
            right = df[curr].fillna("").astype(str)
            df[prev] = (
                (left + " " + right)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
                .replace({"": pd.NA})
            )
            df = df.drop(columns=[curr])
            cols.pop(i)
            continue
        i += 1

    drop_cols = [
        c
        for c in df.columns
        if _is_auto_col(c) and (df[c].isna().all() or (df[c].astype(str).str.strip() == "").all())
    ]
    return df.drop(columns=drop_cols) if drop_cols else df


def ods_table(path: Path) -> pd.DataFrame:
    doc = load(str(path))
    tables = doc.getElementsByType(Table)
    if not tables:
        raise ValueError(f"No se encontraron tablas en {path.name}")

    rows: list[list[str]] = []
    for r in tables[0].getElementsByType(TableRow):
        cells = extract_row_cells(r)
        if any(_clean_cell(c) for c in cells):
            rows.append(cells)

    if not rows:
        raise ValueError(f"Tabla vacía en {path.name}")

    header_idx = find_header_idx(rows)
    header = normalize_header(CUSTOM_HEADER_ODT if header_idx is None else rows[header_idx])
    data = rows if header_idx is None else rows[header_idx + 1 :]

    return fix_merged_header_columns(build_df_from_rows(data, header))


# ---------- pdf ----------
def pdf_table(path: Path) -> pd.DataFrame:
    dfs = tabula.read_pdf(str(path), pages="all", lattice=True, multiple_tables=True)
    if not dfs:
        raise ValueError(f"No se detectaron tablas en {path}")

    header_cols = normalize_header(list(dfs[0].columns))
    if not is_header_row([header_cols[0] if header_cols else ""]):
        header_cols = CUSTOM_HEADER_PDF  # ya la tenés con el largo correcto

    merged = []
    for idx, df in enumerate(dfs):
        df = df.copy()
        if idx > 0:
            # Mantengo tu lógica tal cual
            new_first_row = list(df.columns)
            df = df.reset_index(drop=True)
            df.loc[-1] = new_first_row
            df.index = df.index + 1
            df = df.sort_index().reset_index(drop=True)

        df.columns = header_cols
        merged.append(df.dropna(how="all").reset_index(drop=True))

    return clean_x000d_df(pd.concat(merged, ignore_index=True))


# ---------- salida ----------
def save_table(df: pd.DataFrame, output_dir: Path, base_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = df.astype("string")
    print(f"-> Guardando como: {base_name}.parquet y {base_name}.xlsx")
    df.to_parquet(output_dir / f"{base_name}.parquet")
    df.to_excel(output_dir / f"{base_name}.xlsx", index=False)


def extract_table(path: Path, output_dir: Path) -> None:
    ext = path.suffix.lower()
    base_name = path.stem

    if ext == ".pdf":
        df = pdf_table(path)
    elif ext in {".ods", ".odt"}:
        df = ods_table(path)
    else:
        raise ValueError(f"Extensión no soportada para extracción tabular: {ext}")

    save_table(clean_x000d_df(df), output_dir, base_name)
