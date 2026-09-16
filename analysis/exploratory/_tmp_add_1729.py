import json
nb = json.load(open('analysis/prediction_models_v2.ipynb', encoding='utf-8'))

ids = [c.get('id') for c in nb['cells']]
idx = ids.index('VSCfo04')

def to_source(text):
    lines = text.split('\n')
    return [ln + '\n' for ln in lines[:-1]] + [lines[-1]]

md_cell = {
    'cell_type': 'markdown',
    'id': 'VSCfo05md',
    'metadata': {},
    'source': to_source(
        "### 17c. Single Pre-Registered Confirmatory Test\n"
        "\n"
        "Per thesis H2 (conversation-structure/participation features relate to group-functioning "
        "outcomes), this tests **one** hypothesis chosen a priori, not selected from the 17b scan: "
        "does the others' mean backchannel rate predict a focal participant's `voice_inclusion` rating, "
        "beyond task + group intercept? No FDR correction is applied because this is a single test."
    ),
}

code_cell = {
    'cell_type': 'code',
    'id': 'VSCfo05',
    'metadata': {},
    'execution_count': None,
    'outputs': [],
    'source': to_source(
        "res = fit_mixedlm_pair(fo_model_df, 'voice_inclusion', ['others_mean_tr_backchannel_count'],\n"
        "                        ['T1', 'T2', 'T3'])\n"
        "if res is None:\n"
        "    print('Insufficient data for this test.')\n"
        "else:\n"
        "    coef, p = res['feat_coefs']['others_mean_tr_backchannel_count']\n"
        "    print('H: others_mean_tr_backchannel_count -> voice_inclusion')\n"
        "    print(f\"n={res['n']}  n_groups={res['n_groups']}  ICC={res['icc']}\")\n"
        "    print(f\"LR_stat={res['lr_stat']}  dof={res['dof']}  LR_p={res['lr_p']}\")\n"
        "    print(f\"coef={coef}  coef_p={p}  converged={res['converged']}  n_per_param={res['n_per_param']}\")\n"
    ),
}

nb['cells'][idx+1:idx+1] = [md_cell, code_cell]

with open('analysis/prediction_models_v2.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write('\n')

print('Inserted cells after', idx, '-> new total:', len(nb['cells']))
