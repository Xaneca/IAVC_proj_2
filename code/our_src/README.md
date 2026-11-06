# Codigo do grupo

A Stage A e o baseline SVM estao organizados em
[`part1/`](part1/stagea/README.md). Configuracao atual:
`part1/stagea/config.json`.

Na pasta `code/`:

```powershell
python -m our_src.part1.stagea.stage_a
```

Resultados em `results/part1/stageA/centroids`, `results/part1/stageA/metrics.csv` e
`results/part1/stageA/svm_ap.csv`.
SIFT e histogramas sao mantidos apenas em RAM. O teste final fica reservado.

`make_splits.py` reproduz os splits globais por imagem.
`artifact_json.py` contem utilitarios JSON partilhados por todas as stages.
`part1/stagea/metrics_utils.py` contem metricas auxiliares exclusivas da Stage A.
As configuracoes anteriores em `our_src/configs` nao sao usadas pela nova Stage A.

## Parte 1 — A-G

Guia em [`part1/`](part1/README.md); executar `python -m our_src.part1.run --dry-run` e
depois `python -m our_src.part1.run` a partir de `code/` (ou `run_part_1.ps1` em Windows).

## Parte 3

Guia em [`part3/`](part3/README.md); executar `.\run_part_3.ps1` a partir de `code/`.

## Parte 4 — representações aprendidas

Código e configuração em [`part4/`](part4/README.md).
A partir da raiz Git: `python code/run_deepmodels.py --help`.
Guia geral: [`PART4.md`](../../PART4.md).

## Parte 2 — H/I

Código em [part2/](part2/README.md); executar `python run_part_2.py` a partir de `code/`.
