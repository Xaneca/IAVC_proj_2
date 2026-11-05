# Codigo do grupo

A Stage A e o baseline SVM estao organizados em [`part1/`](part1/README.md).
Configuracao atual: `part1/stage_a.json`.

Na pasta `code/`:

```powershell
python -m our_src.part1.stage_a
```

Resultados em `stageA/centroids`, `stageA/metrics.csv` e `stageA/svm_ap.csv`.
SIFT e histogramas sao mantidos apenas em RAM. O teste final fica reservado.

`make_splits.py` reproduz os splits globais por imagem. `a_json.py` e
`a_metrics.py` contem utilitarios partilhados, reutilizados por `part1`.
`python -m our_src.stage_a` continua como alias para treinar apenas vocabularios.
As configuracoes anteriores em `our_src/configs` nao sao usadas pela nova Stage A.
