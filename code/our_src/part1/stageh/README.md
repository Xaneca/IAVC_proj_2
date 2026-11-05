# Stage H — BoVW, VLAD e Fisher

Instalar esta pasta em `code/our_src/part1/stageh/`. Executar no ambiente do projeto:

```powershell
python -m our_src.part1.stageh.stage_h
python -m our_src.part1.stageh.stage_h --smoke-per-class 2
```

| Experiência | Encoder | K | Normalização | Dimensão |
|---|---|---:|---|---:|
| E1 | BoVW | 64 | L2 | 64 |
| E2 | VLAD | 64 | Intra → L2 | 8192 |
| E3 | Fisher | 64 | Signed power → L2 | 16448 |
| E4 | BoVW | 128 | L2 | 128 |
| E5 | VLAD | 128 | Intra → L2 | 16384 |
| E6 | Fisher | 128 | Signed power → L2 | 32896 |

São 30 modelos finais: seis experiências × cinco classes (motorbike,
aeroplane, horse, car e person), cada uma contra background. Encoding global,
SIFT de 128 dimensões, sem PCA. Mantém exatamente os splits históricos.

Na experiência completa, reutiliza os K-means K=64/128 de `results/part1/stageA/centroids`,
validando treino, extrator e imagens. Se faltarem, calcula-os com o código de A.
BoVW e VLAD partilham esses centroides. O Fisher ajusta dois GMMs diagonais à
mesma amostra determinística de SIFT do treino usada nos vocabulários: mesma
lista de imagens, ordem, seed e limite de 400 descritores/imagem. Os dois
K-means têm de partilhar esse protocolo. Os GMMs têm random_state=42 explícito,
sem recorrer ao ramo GMM não determinístico de `src.train.train_encoder`.
GMMs não convergidos causam erro; os parâmetros são configuráveis em `gmm`.

Os adaptadores em `encoders.py` reutilizam as classes fornecidas em
`src.encoders`: BoVW diretamente; VLAD com `normalize_components=True`, sem
power nem mass normalization; Fisher com signed square root, alpha=0.5,
seguida de L2 segura. Vetores nulos permanecem nulos. O Fisher fornecido por
skimage inclui estatísticas dos pesos: dimensão K(2D+1), em vez de 2KD.
Não é necessário alterar `src`, preservando a identidade dos modelos antigos.

Reutiliza `train_svm_grid` e `src.utils.train_svm`. Cada experiência/classe
seleciona C=[0.1,1,10] por AP de validação, sem nova normalização no SVM.
Todos os parâmetros são congelados antes da avaliação no teste. Não volta a
treinar na validação. A escolha entre as seis experiências também usa AP de
validação, com primeiro vencedor em caso de empate.

BoVW E1/E4 são recalculados com o mesmo protocolo desta comparação; os
centroides são reutilizados, mas não se misturam métricas ou pesquisas de C
de execuções anteriores. Isso evita usar resultados com dados/configurações
diferentes como baseline.

Saídas completas em `results/part1/stageH/`; smoke em `results/part1/stageH/smoke/`:

- `encoders/kmeans_references.json`: referências e SHA256 dos centroides.
- `encoders/gmm_*.npz`: médias, covariâncias, priors, seed, amostra, tempos e
  parâmetros de treino. Ficheiros sem pickle, reutilizados quando compatíveis.
- `models/<classe>__<E1…E6>.npz`: 30 SVMs, pesos, bias e metadados de inferência.
- `metrics.csv`: AP/precision/recall/F1/accuracy/TN/FP/FN/TP por train/val/test,
  K, dimensão, C, tempo de ajuste do encoder e SVM, inferência, encoding e
  extração SIFT. A coluna `selected_by_validation` identifica o vencedor da classe.
- `validation_trials.csv`: todos os candidatos de C, sem resultados de teste.
- `curves/<classe>/<experiência>/<split>/`: PR em CSV/PNG e AP em JSON.
- `rankings/<classe>/<experiência>/<split>.csv`: IDs, scores e labels ordenados.
- `run.json`: configuração, estado, protocolo e SHA256 dos splits.

Precision e afins usam threshold SVM 0. Tempos de SIFT e encoding são partilhados
entre classificadores: não os somar nas 30 linhas. O tempo de encoder pode ser
histórico; quando não foi registado é vazio. O tempo de inferência inclui a
montagem dos vetores do split e o cálculo dos scores.

Uma passagem de encoding extrai SIFT uma vez por imagem e calcula as seis
representações. As passagens de amostragem para ajustar encoders são separadas.
SIFT fica apenas em RAM. As matrizes de encodings usam ficheiros temporários
para limitar RAM, removidos no fim. Não se guardam descritores SIFT em disco.

No smoke, seleciona duas imagens por categoria/split (a união pode incluir mais
de duas imagens de algumas classes multilabel), mantendo os splits originais.
K-means e GMM usam apenas as imagens de treino dessa união, em
`results/part1/stageH/smoke/encoders/`, sem depender de A concluído nem alterar seus centroides.
Mantém K=64/128 e gera as mesmas 30 combinações, sem reduzir a tabela a K=32.

Para inferência, carregar os arrays com `allow_pickle=False`, construir o encoder
com os parâmetros referidos no modelo, aplicar `encode` e calcular `x @ w + b`.
As referências a modelos são caminhos locais; ao mover resultados, atualizar
os caminhos ou resolver os ficheiros pelo SHA256.

Testes: `python -m unittest our_src.part1.stageh.test_stage_h`.
Pode correr H isoladamente; não exige terminar B–G.
O `run_all.ps1` também inclui H no fim da cadeia. Para H apenas:
`.\run_all.ps1 -Start H -End H`. O modo `-Smoke` da cadeia inclui H.
