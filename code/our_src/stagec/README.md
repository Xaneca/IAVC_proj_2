# Stages C e D

Instalação no projeto original (pasta `code/`): copiar `stage_c_setup/` para
`our_src/stagec/` e `stage_d_setup/` para `our_src/staged/`. Atualizar também
`our_src/part1/data.py` com `part1_setup/data.py`, e os módulos antigos
`our_src/centroids_stage_a.py` e `our_src/stage_a.py` com, respetivamente,
`stage_a_setup/centroids_stage_a.py` e `stage_a_setup/stage_a_v2.py` se os usares:
assim A, B, C e D validam
as mesmas listas históricas. Não voltar a gerar splits. A referência guardada
em `stagec/split_reference/` contém as listas de A/B (1978 treino, 494 validação,
2474 teste), incluindo todas as categorias. Qualquer alteração de pertença,
categoria ou partição causa erro. A ordem das linhas não altera o split.

É necessário terminar as execuções completas de A e B primeiro. O código
recusa resultados smoke/cancelados de B. Usa o mesmo vocabulário e encoding
selecionados em B, validando a identidade dos centroides e os IDs de treino.

Na pasta `code/`, no ambiente do projeto:

```powershell
python -m our_src.stagec.stage_c
python -m our_src.staged.stage_d
python -m unittest our_src.stagec.test_evaluation
```

C avalia motorbike/background com o modelo final de B, sem novo treino.
D reutiliza `stagec.stage_c.run` e `stagec.evaluation.evaluate` para
aeroplane/background, horse/background, car/background e person/background.
Em D cada SVM é treinado só no treino e C é selecionado pelo AP de validação.
Nenhuma etapa escolhe parâmetros usando o teste nem volta a treinar com validação.
Os histogramas são reutilizados na cache de B; não se guardam descritores SIFT.

Cada saída `results/stageC/` ou `results/stageD/` contém:

- `run.json`: configuração, estado e SHA256 dos splits.
- `split_manifest.csv`: IDs, classes e partições efetivamente usados.
- `aps_by_class.csv` e `aps_by_class.json`: métricas por classe e split.
- `<classe>/svm_model.npz`: pesos, bias e metadados, sem pickle.
- `<classe>/validation_trials.csv`: pesquisa de C em D.
- `<classe>/<train|val|test>/precision_recall.csv`: precision, recall e threshold.
- `<classe>/<train|val|test>/precision_recall.png`: curva PR.
- `<classe>/<train|val|test>/ap.json`: AP num ficheiro separado.
- `<classe>/<train|val|test>/ranking.csv`: ranking decrescente, score contínuo,
  label, ID, classe e split; empates ordenados por ID.

O último ponto da curva não tem threshold (campo vazio), conforme sklearn.
AP usa average_precision_score, a soma dos incrementos de recall ponderados
pela precisão, e não a integração trapezoidal. Os scores SVM não são probabilidades.
Para inferência, extrair o encoding descrito nos metadados, aplicar a sua
normalização uma vez e calcular `x @ weights + bias`.

Interpretou-se o pedido “svm” para precision/recall como CSV.

Nesta pasta de preparação, os testes de métricas e splits também correm com
`python -m unittest stage_c_setup.test_evaluation` (numpy, sklearn, matplotlib).
