from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import colorsys

# Matches: color = [mode]? { v1 v2 v3 [v4] } with ints/floats, commas or spaces, multiline ok
COLOR_RE = re.compile(
    r'\bcolor\s*=\s*(?:([a-zA-Z]+)\s*)?\{\s*'
    r'([+-]?\d+(?:\.\d+)?)\s*[, ]+'
    r'([+-]?\d+(?:\.\d+)?)\s*[, ]+'
    r'([+-]?\d+(?:\.\d+)?)'
    r'(?:\s*[, ]+([+-]?\d+(?:\.\d+)?))?\s*\}',
    re.IGNORECASE | re.DOTALL
)

def _to_hex_rgb(r: float, g: float, b: float) -> str:
    # clamp and round to 0..255 then format #RRGGBB
    ri = max(0, min(255, int(round(r))))
    gi = max(0, min(255, int(round(g))))
    bi = max(0, min(255, int(round(b))))
    return f"#{ri:02X}{gi:02X}{bi:02X}"

def _hsv_triplet_to_rgb(v1: float, v2: float, v3: float) -> tuple[float,float,float]:
    h, s, v = v1, v2, v3
    # Normalize common notations:
    # If H looks fractional, assume 0..1; else if up to 360, scale down.
    if h <= 1.0:
        h = h * 360.0
    if h > 360.0:
        h = h % 360.0
    # If S/V look like percentages (e.g., 50, 75), convert to 0..1
    if s > 1.0: s = s / 100.0 if s <= 100.0 else 1.0
    if v > 1.0: v = v / 100.0 if v <= 100.0 else 1.0
    # colorsys expects h in 0..1
    r, g, b = colorsys.hsv_to_rgb(h/360.0, max(0,min(1,s)), max(0,min(1,v)))
    return (r*255.0, g*255.0, b*255.0)

def extract_color_from_text(text: str) -> str | None:
    # Remove comments starting with # (whole line or inline)
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        cleaned_lines.append(line.split("#", 1)[0])
    cleaned = "\n".join(cleaned_lines)

    m = COLOR_RE.search(cleaned)
    if not m:
        return None

    mode = (m.group(1) or "rgb").lower()  # default to rgb if omitted
    v1 = float(m.group(2)); v2 = float(m.group(3)); v3 = float(m.group(4))
    # group(5) could be alpha; we ignore it

    if mode in ("rgb", "rgba"):
        return _to_hex_rgb(v1, v2, v3)
    elif mode == "hsv":
        r, g, b = _hsv_triplet_to_rgb(v1, v2, v3)
        return _to_hex_rgb(r, g, b)
    else:
        # Unknown mode; try to interpret as rgb as a fallback
        return _to_hex_rgb(v1, v2, v3)

def extract_color_from_file(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"Error reading {path}: {e}", file=sys.stderr)
        return None
    return extract_color_from_text(text)

def main():
    PATH = "countries"
    OUT = "python/geodata/colors.json"
    root = Path(PATH)
    if not root.exists() or not root.is_dir():
        print(f"Input directory not found: {root}", file=sys.stderr)
        sys.exit(1)

    files = (root.glob("*.txt"))
    mapping: dict[str, str] = {}
    total_files = 0

    for fp in files:
        total_files += 1
        hex_color = extract_color_from_file(fp)
        if not hex_color:
            # Not found: warn, but continue
            print(f"Warning: no color found in {fp}", file=sys.stderr)
            continue

        name = fp.stem
        #name = name.replace("_", " ").strip()

        mapping[name] = hex_color

    # Write JSON
    try:
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error writing {OUT}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Wrote {len(mapping)} entries (scanned {total_files} files) to {OUT}")

if __name__ == "__main__":
    main()
