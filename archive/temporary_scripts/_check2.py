import json, pathlib
nb = json.loads(pathlib.Path('analysis/effectiveness_prediction_models.ipynb').read_text(encoding='utf-8'))
print(f'Total cells: {len(nb["cells"])}')
code_n = 0
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if cell["cell_type"] == "code":
        code_n += 1
        defines = []
        if "model_df2 =" in src: defines.append("model_df2")
        if "model_df3 =" in src: defines.append("model_df3")
        if "model_df " in src or "model_df=" in src: defines.append("model_df")
        uses = []
        if "model_df2" in src and "model_df2 =" not in src: uses.append("uses model_df2")
        if "model_df3" in src and "model_df3 =" not in src: uses.append("uses model_df3")
        print(f"Cell {i} (code#{code_n}): defines={defines} {uses}  src[:50]={repr(src[:50])}")
