import pathlib, unicodedata

path = pathlib.Path("test_case_sensitivity.py")
for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
    if '\u00A0' in line:
        print(f"{lineno:>4}: {line.replace('\u00A0', '⍟')}") 