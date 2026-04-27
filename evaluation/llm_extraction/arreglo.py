from pathlib import Path
import unicodedata
import sys


def fix_unicode_file(file_path: str) -> None:
    path = Path(file_path)

    text = path.read_text(encoding="utf-8")

    normalized = unicodedata.normalize("NFC", text)

    if text == normalized:
        print("No se encontraron caracteres Unicode combinados.")
        return

    print(f"\n[CAMBIOS EN] {path}")

    original_lines = text.splitlines()
    fixed_lines = normalized.splitlines()

    for i, (orig, fixed) in enumerate(zip(original_lines, fixed_lines), start=1):
        if orig != fixed:
            print(f"\nLínea {i}")
            print(f"ANTES : {orig}")
            print(f"DESPUÉS: {fixed}")

    path.write_text(normalized, encoding="utf-8")

    print("\nArchivo corregido correctamente.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python fix_unicode.py archivo.txt")
    else:
        fix_unicode_file(sys.argv[1])