# Stage G — curvas de aprendizagem

Copiar para `code/our_src/stageg/`, com F instalado e completo. Executar em `code/`:

```powershell
python -m our_src.stageg.stage_g
```

Motorbike/background. Compara a configuração linear escolhida em E com a
melhor variante Hellinger escolhida em F por AP de validação. Mantém K, tiling,
transformações e parâmetros SVM descobertos anteriormente. Reajusta apenas os
pesos e bias do SVM para cada subconjunto; não faz nova seleção de C ou kernel.

Defaults configuráveis: frações [0.10, 0.25, 0.50, 0.75, 1.00], dez seeds
distintas [0,…,9]. São 100 treinos (5 frações × 10 seeds × 2 kernels).
`StratifiedShuffleSplit` seleciona apenas IDs da partição interna train,
preservando as proporções motorbike/background aproximadamente. Para cada
par fração/seed, os dois kernels recebem exatamente os mesmos IDs. Diferentes
frações não são necessariamente subconjuntos encaixados. O tamanho é
`floor(fração * N_train)`; a fração 1 usa todo o treino. Frações que não
permitem amostrar as duas classes são rejeitadas. Validação e teste não são
subamostrados, e nunca contribuem para o ajuste do SVM.

O vocabulário permanece o aprendido no treino completo das etapas anteriores:
esta é uma curva de aprendizagem do classificador, não da pipeline inteira.
O subconjunto não desencadeia novo K-means ou seleção por validação/teste.

- `results/stageG/metrics.csv`: uma linha por kernel/fração/seed, AP e restantes métricas
  de validação e teste, número de imagens de treino por classe, C, operações,
  SHA256 dos splits e tempos de treino/inferência.
- `results/stageG/metrics_summary.csv`: AP de teste médio e desvio padrão amostral entre
  seeds, e AP de validação médio, por fração/kernel.
- `results/stageG/ap_vs_train_images.png`: AP de teste médio versus número real de imagens
  de treino (positivas + background), com barras de ±1 desvio padrão.
- `results/stageG/train_subsets.json`: IDs exatos usados por fração/seed.
- `results/stageG/run.json`: estado, configuração e proveniência dos modelos E/F.

O vocabulário, parâmetros e variantes são congelados antes destas experiências.
Não escolher a fração ou reajustar parâmetros pelo gráfico de teste. Um CSV
parcial só representa as experiências já terminadas, não um run completo.
