# Stage E

Copiar esta pasta para `code/our_src/stagee/`. Requer os módulos anteriores,
incluindo `stagec/protocol.py` e `stagec/split_reference/`. No ambiente do projeto,
na pasta `code/`:

```powershell
python -m our_src.stagee.stage_e
```

Para cada classe (motorbike, aeroplane, horse, car, person), sempre contra
background, guarda nove SVMs: L1/L2/sem normalização × global/2x2/pirâmide.
Global tem K componentes, 2x2 tem 4K, pirâmide tem 21K (1x1 + 2x2 + 4x4),
com os pesos espaciais das etapas anteriores. A normalização é aplicada uma vez
ao vetor completo, depois dos pesos espaciais. “none” não aplica normalização.

O K e os centroides são os mesmos de B, para comparar apenas normalização e
tiling. Reutiliza a cache raw de B sem extrair SIFT por configuração. Recusa
execuções incompletas ou smoke de A/B e alterações aos splits históricos.

Cada combinação seleciona C=[0.1,1,10] pelo AP de validação, usando as funções
anteriores. São nove modelos finais por classe (45 no total); os candidatos
de C são treinos de seleção. Não se treina na validação nem se seleciona pelo
teste. O teste é avaliado para cada modelo depois da seleção do respetivo C.

- `results/stageE/metrics.csv`: 45 linhas, uma por classe/combinação, com AP de treino,
  validação e teste, precision/recall/F1/accuracy de teste no threshold SVM 0,
  C selecionado, dimensão, K, tempo de treino do candidato selecionado,
  números de imagens, SHA256 dos splits e caminho do modelo.
- `results/stageE/models/<classe>__<global|2x2|pyramid>__<l1|l2|none>.npz`:
  pesos, bias e metadados do encoding, vocabulário, splits e parâmetros.
- `results/stageE/run.json`: configuração, proveniência e estado. Um CSV parcial durante
  uma execução interrompida não representa uma experiência completa.

Carregar NPZ com `allow_pickle=False`. Para inferência, derivar o vetor com
`stageb.histograms.svm_vector` usando K, layout e normalization dos metadados,
e calcular `x @ weights + bias`.

Testes locais: `python -m unittest stage_e_setup.test_experiment`.
No projeto instalado: `python -m unittest our_src.stagee.test_experiment`.
