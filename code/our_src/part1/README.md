# Part 1 — Stage A e baseline SVM

Executar a partir de `code/`, no ambiente do projeto.

```powershell
python -m our_src.part1.stage_a
```

Configuracao: `our_src/part1/stage_a.json`. Uma seed (42), K=32,64,128,256,512,1024.
O comando calcula automaticamente os centroides em falta antes do encoding e
das métricas. O K-means usa apenas o treino interno, uma amostra determinística
de até 400 SIFT por imagem, partilhada entre os K em falta. Centroides existentes
são validados e reutilizados; modelos incompatíveis causam erro explícito.
Não é necessário executar previamente `vocabulary` nem recuperar modelos de Git.
Os centroides e o tempo de ajuste ficam guardados em JSON. Os SIFT ficam em RAM.
No modo smoke, os centroides são aprendidos só nas imagens de treino selecionadas
e ficam em `stageA/smoke/centroids`, sem misturar com a experiência completa.
Resultados em `code/stageA/`:

```text
stageA/
  centroids/       # vocabularios partilhados entre layouts
  metrics.csv
  svm_ap.csv
  run.json        # protocolo, configuracao, versoes, tempos e melhores AP de validacao
  logs/
```

O pull removeu os resultados antigos. Para os relocalizar explicitamente do
historico local, sem reajustar modelos:

```powershell
python -m our_src.part1.stage_a --recover-centroids-from 0ea096a
```

O caminho antigo permanece removido. Tambem se podem treinar centroides novos:
`python -m our_src.part1.vocabulary`. Os centroides carregados sao verificados
contra o treino interno, o extrator e as imagens usadas no ajuste.

## Layouts e reutilizacao

| layout | regioes | dimensao |
|---|---|---:|
| normal / pyramid_L0 | 1x1 | K |
| flat_2x2 | 2x2 | 4K |
| pyramid_L1 | 1x1 + 2x2 | 5K |
| pyramid_L2 | 1x1 + 2x2 + 4x4 | 21K |

SIFT e extraido uma vez por imagem, usando `src.features.extractFeatures` sem
cache em disco. Cada K usa `src.encoders.KMeansPredictor.predict_transform`
uma vez por batch, independentemente do tiling. `src.encoders.spatial_tiling`
seleciona as 16 regioes finais; somam-se celulas adjacentes para obter 2x2.
`src.encoders.removeSpatialInformation` recupera o global. Celulas vazias ficam
com zeros. Normal e pyramid_L0 partilham a mesma matriz em RAM.

Os SIFT e atribuicoes sao descartados por imagem; os histogramas compactos
uint32 ficam apenas em RAM ate terminar metricas/SVM. Nao sao escritos SIFT,
amostra, atribuicoes, histogramas ou modelos SVM em disco. O custo e ter de
reextrair features se se interromper a execucao antes de completar o encoding.

Pesos por nivel da piramide: L0=[1], L1=[1/2,1/2], L2=[1/4,1/4,1/2].
Os pesos sao aplicados antes da normalizacao no baseline; flat nao tem hierarquia.

## Metricas

`metrics.csv` contem uma linha por K/layout/split interno:

- Capacidade nominal: dimensao, celulas, K e dimensao SIFT. Nao e uma prova de AP.
- Erro de quantizacao: media da distancia L2 ao quadrado por descritor,
  independente do layout; tambem a soma e o numero de descritores.
- Tempos de extracao, atribuicao e contagem partilhados; montagem por layout.
  Nao somar as mesmas tarefas partilhadas sobre varias linhas/layouts.
- Ajuste de K-means: null/status explicito se o centroide historico nao guardou
  esse tempo. Nao se inventam tempos nem se repete o treino para os medir.
- Armazenamento: bytes reais do JSON dos centroides, bytes do array float32,
  bytes teoricos dos histogramas uint32 em RAM e zero bytes de features em disco.
- Utilizacao do vocabulario global do split e media por imagem, esparsidade,
  entropia de ocupacao em nats e normalizada por log(K), e palavras efetivas.
- Separabilidade: distancia cosseno media de pares que partilham uma categoria
  (intra) versus pares com categorias disjuntas (inter), com pesos e L2.
  Background e uma categoria. Usa os mesmos ate 5000 pares por split em todos
  os encodings. Reporta os numeros de pares, exclusoes de vetores zero e inter-intra.
- Estabilidade entre seeds: nao avaliada com a unica seed pedida, registada
  explicitamente em CSV. Nao confundir estabilidade com qualidade do vocabulario.

Nao e gerado grafico cotovelo.

## SVM generico

`our_src.part1.svm.train_svm_grid(features, train, validation, param_grid,
test=None, seed=42)` recebe qualquer encoding com dimensao fixa:

```python
features = {"imagem1": vetor1, "imagem2": vetor2}  # todos os splits necessarios
train = {"imagem1": 1, ...}                       # labels -1/+1
validation = {"imagem2": -1, ...}
result = train_svm_grid(features, train, validation,
                       {"C": [0.1, 1, 10], "normalization": ["l2"]})
```

Treina atraves de `src.utils.train_svm` e seleciona o maior AP de validacao,
calculado com scores continuos. Aceita grelhas C, normalization (none/l1/l2),
max_iter e tol. Proibe IDs partilhados entre splits e exige ambas as classes.
Se receber `test`, so avalia o candidato selecionado, uma vez, depois da escolha;
nao re-treina com validacao. Nao usar resultados de test para comparar variantes.
A funcao fornecida oculta avisos de convergencia: max_iter=20000 nao e garantia
de convergencia, e os resultados nao declaram esse diagnostico.

O baseline default e motorbike positivo versus background negativo (Stage B),
todos os K/layouts, C=[0.1,1,10], L2, tol=1e-4, max_iter=20000.
Pode-se configurar `baseline_categories` para outras categorias.
`svm_ap.csv` guarda todos os candidatos, AP de treino/validacao, tempos e
o candidato selecionado dentro de cada encoding. `run.json` resume o melhor
K/layout/C por categoria. O teste final fica reservado; nao e carregado.

## Verificacao e execucao pequena

```powershell
python -m unittest our_src.part1.test_part1
python -m our_src.part1.stage_a --smoke-per-category 2
```

O smoke usa ate duas imagens por categoria e split e escreve em `stageA/smoke`;
nao substitui a experiencia integral. O processo completo publica os CSV
progressivamente depois da fase de encoding; `run.json` indica running/complete.
