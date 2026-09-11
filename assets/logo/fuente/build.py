"""Genera las piezas del logo aRAGnia.

Uso:
    python3 build.py                     todas las piezas, versión marrón
    python3 build.py --claro             todas, versión clara (para fondos oscuros)
    python3 build.py barra --claro       solo una pieza
    python3 build.py horizontal icono    varias piezas

Requiere macOS: usa qlmanage para rasterizar el SVG.
"""

import argparse
import math
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import alpha_lib
from arana import spider

# --- paletas ---------------------------------------------------------------

PALETA = {
    "marron": dict(texto="#6B3A18", hilo="#C08050", nodo="#A2602C", destacado="#0B5F88"),
    "claro": dict(texto="#FFFFFF", hilo="#E0B48A", nodo="#E0B48A", destacado="#BAE6FD"),
}

FUENTE = (
    '<style>@font-face{font-family:"SFCR";'
    'src:url("file:///System/Library/Fonts/SFCompactRounded.ttf");}</style>'
)

ARANA = spider()

# --- primitivas ------------------------------------------------------------


def bicho(cx, cy, ancho):
    """Coloca la araña centrada en (cx, cy) con el ancho dado."""
    s = ancho / 700.0
    return f'<g transform="translate({cx - 500 * s:.1f},{cy - 465 * s:.1f}) scale({s:.4f})">{ARANA}</g>'


def texto(x, y, tam, anchor, col, espaciado=2):
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'font-family="SFCR, Arial Rounded MT Bold, sans-serif" font-weight="900" '
        f"style=\"font-variation-settings:'wght' 900\" font-size=\"{tam}\" "
        f'fill="{col}" letter-spacing="{espaciado}">'
        f"<tspan>a</tspan><tspan>RAG</tspan><tspan>nia</tspan></text>"
    )


def tela(cx, cy, radios, angulos, r_nodo, destacados, grosor, p, cerrada=False,
         panza=0.90, solo_destacados=False):
    """Dibuja la telaraña. Los nodos destacados van al doble de tamaño y en otro color."""

    def punto(a, r):
        return (cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a)))

    paso = 360.0 / len(angulos)
    hilos = [
        f'<line x1="{cx}" y1="{cy}" x2="{punto(a, radios[-1])[0]:.1f}" '
        f'y2="{punto(a, radios[-1])[1]:.1f}"/>'
        for a in angulos
    ]
    for r in radios:
        tramos = []
        indices = range(len(angulos)) if cerrada else range(len(angulos) - 1)
        for i in indices:
            a1 = angulos[i]
            a2 = angulos[(i + 1) % len(angulos)]
            # al cerrar el círculo, el punto de control avanza medio paso desde a1:
            # promediar a1 y a2 daría el lado opuesto de la tela
            medio = a1 + paso / 2 if cerrada else (a1 + a2) / 2
            x1, y1 = punto(a1, r)
            x2, y2 = punto(a2, r)
            qx, qy = punto(medio, r * panza)
            tramos.append(f"M {x1:.1f} {y1:.1f} Q {qx:.1f} {qy:.1f} {x2:.1f} {y2:.1f}")
        hilos.append(f'<path d="{" ".join(tramos)}"/>')

    nodos = ""
    for i, r in enumerate(radios):
        for j, a in enumerate(angulos):
            grande = (i, j) in destacados
            if not grande and solo_destacados:
                continue
            x, y = punto(a, r)
            radio = r_nodo * 2.1 if grande else r_nodo
            col = p["destacado"] if grande else p["nodo"]
            nodos += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radio:.1f}" fill="{col}"/>'

    return (
        f'<g fill="none" stroke="{p["hilo"]}" stroke-width="{grosor}" '
        f'stroke-linecap="round">{"".join(hilos)}</g><g>{nodos}</g>'
    )


def envolver(lado, cuerpo, fondo):
    # la fuente solo se declara si la pieza lleva texto: así el SVG del ícono
    # y del favicon queda autocontenido y se ve igual fuera de macOS
    fuente = FUENTE if "<text" in cuerpo else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{lado}" height="{lado}" '
        f'viewBox="0 0 {lado} {lado}">{fuente}'
        f'<rect width="{lado}" height="{lado}" fill="{fondo}"/>{cuerpo}</svg>'
    )


# --- piezas ----------------------------------------------------------------

A12 = [i * 30 for i in range(12)]
A8 = [i * 45 for i in range(8)]
ABANICO = [18 + i * (144 / 8) for i in range(9)]
DESTACADOS = {(3, 1), (2, 4), (3, 6), (2, 10)}


def horizontal(p):
    lado, cy, r, cx = 1700, 850, 250, 520
    cuerpo = (
        tela(cx, cy, [r * 0.40, r * 0.60, r * 0.80, r], A12, 12, DESTACADOS, 6, p, cerrada=True)
        + bicho(cx, cy - 20, 385)
        + texto(cx + r + 18, cy + 62, 180, "start", p["texto"])
    )
    return lado, cuerpo


def vertical(p):
    cuerpo = (
        tela(600, 490, [130, 215, 300, 372], ABANICO, 11, {(3, 1), (2, 4), (3, 6), (1, 7)},
             6, p, cerrada=False, panza=0.86)
        + bicho(600, 305, 600)
        + texto(600, 1055, 170, "middle", p["texto"])
    )
    return 1200, cuerpo


def icono(p):
    c, r = 500, 430
    cuerpo = (
        tela(c, c, [r * 0.40, r * 0.60, r * 0.80, r], A12, 17, DESTACADOS, 9, p, cerrada=True)
        + bicho(c, c - 35, 620)
    )
    return 1000, cuerpo


def favicon(p):
    """Versión simplificada: menos anillos e hilos más gruesos, legible a 32 px."""
    c, r = 500, 470
    cuerpo = (
        tela(c, c, [r * 0.66, r], A8, 24, {(1, 1), (0, 4), (1, 6)}, 30, p,
             cerrada=True, solo_destacados=True)
        + bicho(c, c - 10, 800)
    )
    return 1000, cuerpo


def barra(p):
    """Lockup de la topbar: el texto pesa más que el símbolo para leerse a poca altura."""
    lado, cy, r, cx = 1700, 850, 175, 330
    cuerpo = (
        tela(cx, cy, [r * 0.40, r * 0.60, r * 0.80, r], A12, 9, DESTACADOS, 5, p, cerrada=True)
        + bicho(cx, cy - 14, 270)
        + texto(cx + r + 26, cy + 66, 205, "start", p["texto"])
    )
    return lado, cuerpo


PIEZAS = {
    "horizontal": (horizontal, 2400),
    "logo": (vertical, 2400),
    "icono": (icono, 1600),
    "favicon": (favicon, 1024),
    "barra": (barra, 1800),
}

# --- render ----------------------------------------------------------------


def generar(nombre, claro):
    dibujar, resolucion = PIEZAS[nombre]
    p = PALETA["claro" if claro else "marron"]
    salida = f"aragnia-{nombre}" + ("-claro" if claro else "")

    # Se renderiza sobre blanco y sobre negro; comparando ambos se recupera
    # el alfa real, porque qlmanage siempre rasteriza con fondo opaco.
    for fondo, sufijo in (("#FFFFFF", "_w"), ("#000000", "_b"), ("none", "")):
        lado, cuerpo = dibujar(p)
        with open(f"{salida}{sufijo}.svg", "w") as f:
            f.write(envolver(lado, cuerpo, fondo))

    for sufijo in ("_w", "_b"):
        png = f"{salida}{sufijo}.svg.png"
        if os.path.exists(png):
            os.remove(png)
        subprocess.run(
            ["qlmanage", "-t", "-s", str(resolucion), "-o", ".", f"{salida}{sufijo}.svg"],
            capture_output=True,
        )

    alpha_lib.compose(f"{salida}_w.svg.png", f"{salida}_b.svg.png", f"{salida}.png")
    ancho, alto = alpha_lib.crop(f"{salida}.png", 16)
    print(f"{salida}  {ancho}x{alto}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("piezas", nargs="*", choices=list(PIEZAS) + [], default=None,
                    help="piezas a generar (por defecto: todas menos barra)")
    ap.add_argument("--claro", action="store_true",
                    help="versión clara, para fondos oscuros")
    args = ap.parse_args()

    piezas = args.piezas or ["horizontal", "logo", "icono", "favicon"]
    for nombre in piezas:
        generar(nombre, args.claro)


if __name__ == "__main__":
    main()
