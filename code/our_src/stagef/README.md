# Stage F — motorbike/background, Hellinger

Copiar esta pasta para `code/our_src/stagef/`. Requer as etapas anteriores e
Stage E completo. Na pasta `code/`, no ambiente do projeto:

```powershell
python -m our_src.stagef.stage_f
```

“Motorcycle” corresponde à categoria `motorbike` dos splits. Escolhe a melhor
das nove configurações desta classe em E pelo AP de validação, nunca pelo teste.
Reutiliza K, tiling, C, max_iter, tol e seed do modelo E persistido. Não faz grid
search nem treino na validação. Confirma os splits e a identidade do vocabulário.

Compara o modelo linear exato de E e quatro variantes Hellinger:

- `e_norm_sqrt`: melhor normalização de E → raiz.
- `none_sqrt`: sem normalização → raiz.
- `l2_sqrt`: L2 → raiz.
- `sqrt_l2`: raiz → L2.

As operações são aplicadas da esquerda para a direita aos histogramas com os
pesos espaciais do tiling de E. Vetores nulos permanecem nulos. Hellinger usa o
mapa explícito `phi(x)=sqrt(x)` e um SVM sobre esses vetores: o produto interno
é exatamente o kernel aditivo `sum(sqrt(x_i*y_i))`. Não é uma segunda raiz
oculta nem um kernel RBF. Nas variantes, as normalizações anteriores ou
posteriores à raiz fazem parte da definição do kernel. Os quatro nomes são
mantidos mesmo quando a normalização de E faz duas variantes coincidirem.

Saídas:

- `results/stageF/metrics.csv`: cinco linhas com AP, precision, recall, F1, accuracy,
  TN/FP/FN/TP, número de imagens e tempo de inferência, por train/val/test;
  tempo de treino, C e caminho do modelo. Precision e afins usam threshold 0.
- `results/stageF/models/<variante>.npz`: pesos, bias, ordem das operações e proveniência.
- `results/stageF/run.json`: configuração, parâmetros herdados e melhor Hellinger
  escolhido apenas pelo AP de validação; empates preferem a primeira variante.

O tempo de inferência mede os scores do SVM; não inclui SIFT nem transformação.
O baseline copiado de E tem tempo de treino 0 porque não é novamente treinado.
Os valores reais só são produzidos ao correr após concluir E.

Para inferência: carregar com `allow_pickle=False`, obter o histograma do
layout guardado com os pesos espaciais, aplicar `common.transform` seguindo
`metadata.operations`, depois `x @ weights + bias`.

Testes locais: `python -m unittest stage_f_setup.test_experiments`.
No projeto instalado: `python -m unittest our_src.stagef.test_experiments`.
