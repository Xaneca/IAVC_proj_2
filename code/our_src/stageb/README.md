# Stage B — preparar histogramas, SVM e ranking

Codigo em `our_src/stageb/`; resultados em `code/results/stageB/`.
Nao iniciar uma execucao integral sem a pedir explicitamente.

```text
results/stageB/
  run.json
  svm_ap.csv
  svm_model.npz
  histograms/
    metrics.csv
    metrics_definition.json
    <modelo>/<image_id>.npz
  ranking/
    ranking.csv
    relevantVisualWords/<split>_<highest|lowest>_<image_id>.png
```

## Escolha do encoding

`selection.py` exige `results/stageA/run.json` com state=complete e escolhe o maior AP
de validacao da categoria em `results/stageA/svm_ap.csv`. Ignora o teste e os resultados
de smoke. Atualmente a execucao integral da Stage A foi cancelada: por isso
nao existe um vencedor automatico valido. Para testes pequenos usa-se uma
escolha explicita `--k 32 --layout pyramid_L2`; fica marcada como override,
nao como melhor encoding descoberto.

## Histogramas e features do SVM

Guarda-se apenas a piramide raw de tres niveis (1x1 + 2x2 + 4x4), 21K contagens
uint32 por imagem, em NPZ comprimido sem pickle. O ficheiro inclui ID e identidade
dos inputs/configuracao, nunca a imagem ou os SIFT. A cache evita nova extracao
quando se mudam C ou normalizacao; recalcula se mudar a identidade dos inputs.

Dessa unica piramide derivam-se todos os layouts. `svm_vector()` aplica os
pesos por nivel e a normalizacao do encoding escolhido e devolve exatamente
o vetor que entra em `train_svm_grid`. Nao se guarda uma segunda copia do vetor
derivado: a transformacao e deterministica e fica registada em run.json.

Reutiliza `src.features.extractFeatures`, os encoders de `src`,
`our_src.part1.encoding.encode_image` e `our_src.part1.svm.train_svm_grid`,
que chama obrigatoriamente `src.utils.train_svm`.

## Estatisticas por classe

`histograms/metrics.csv` e um CSV longo:

- `record_type=feature`: classe, split, componente, media e variancia populacional
  (ddof=0) dos vetores reais de entrada do SVM.
- `class_summary`: media e variancia da distancia chi-squared de cada imagem
  ao prototipo da sua classe, aprendido apenas no treino interno.
- `class_pair`: distancia chi-squared entre prototipos de classes do treino.

Chi-squared = 0.5 * sum((p-q)^2/(p+q)); termos 0/0 valem 0. Normaliza-se L1
antes da distancia. Uma imagem multilabel contribui para cada classe relevante.
Treino/validacao sao reportados separadamente. O teste final nao e processado.

## SVM e visualizacoes

Motorbike positivo/background negativo por defeito. C=[0.1,1,10], max_iter=20000,
tol=1e-4, seed=42. A normalizacao e a do encoding escolhido. Seleciona o maior
AP de validacao. `svm_ap.csv` guarda os tres candidatos; o modelo selecionado e
guardado em NPZ sem pickle. Nao ha selecao pelo teste final.

O ranking default e do treino: CSV com ID, score continuo, label, split e rank.
Guarda as tres imagens de cada extremo, depois de passarem pela classe fornecida
`src.visualization.RelevantVisualWordsDisplay`, em ranking/relevantVisualWords.

A classe fornecida assume um histograma global Hellinger e dobra os pesos
espaciais numa ordem diferente da concatenacao real. O adaptador em `ranking.py`
agrega as contribuicoes reais por palavra a partir de x*w e compensa esse
Hellinger fixo. Assim, os scores das palavras no display correspondem ao modelo
real. O bias nao e atribuido a palavras visuais. Os PNG mostram o score da imagem
e a palavra positiva mais relevante (highest) ou negativa mais relevante (lowest).

## Apenas testes por agora

Na pasta code/, no ambiente do projeto:

```powershell
python -m unittest our_src.stageb.test_stage_b
python -m our_src.stageb.stage_b --k 32 --layout pyramid_L2 --smoke-per-class 2
```

O smoke usa duas imagens por classe e split e escreve em `results/stageB/smoke`.
Nao confundir a sua AP com resultados integrais. A execucao integral futura
seria `python -m our_src.stageb.stage_b`, apenas depois de completar Stage A.
