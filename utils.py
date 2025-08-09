from pathlib import Path

def save_parser_code(target: str, code: str):
    path = Path("custom_parsers") / f"{target}_parser.py"
    path.parent.mkdir(exist_ok=True)
    path.write_text(code, encoding="utf-8")
    print(f"[INFO] Parser saved at: {path}")
