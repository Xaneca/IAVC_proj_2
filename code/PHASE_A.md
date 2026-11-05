# Fase A — Dense SIFT e BoVW

Executar os comandos na pasta `code/`, com o ambiente `cvc-a2` ativo.
O codigo fornecido permanece em `src/`; a orquestracao fica em
`our_src/part1/`. A configuracao fica em `our_src/part1/stagea/config.json`.

## Etapas

```powershell
python -m our_src.part1.stagea.stage_a --stage inspect
python -m our_src.part1.stagea.stage_a --stage descriptors
python -m our_src.part1.stagea.stage_a --stage sample
python -m our_src.part1.stagea.stage_a --stage vocabularies
python -m our_src.part1.stagea.stage_a --stage encode
```

`inspect` cria um painel de imagens de treino para analisar escala, perspetiva,
iluminacao e fundo. `all` executa as etapas de calculo, sem a inspecao visual.
As etapas seguintes calculam dependencias em falta quando indicado; `encode`
exige os vocabularios ja guardados. Resultados existentes sao reutilizados.

1. **Descritores**: `extractFeatures()` guarda keypoints, SIFT e dimensoes da
   imagem. A mesma cache serve todos os K. Nao aplicar PCA na fase A inicial.
2. **Amostra**: `sampleLocalFeatures()` seleciona descritores apenas do treino.
   Guardar a matriz e as imagens selecionadas; reutilizar a mesma amostra para
   K=256, 512 e 1024. Os limites por classe/imagem sao configuraveis.
3. **Vocabularios**: `train_encoder(..., pca_dim=0)` aprende cada K-means.
   Guardar centroides (`words`), configuracao e tempo. Os NPZ continuam
   compativeis com os encoders fornecidos, podendo ser reutilizados para VLAD
   se o espaco dos descritores e o K forem iguais.
4. **Histogramas**: atribuir cada descritor uma unica vez e contar por regiao
   com `spatial_tiling()`. Guardar as quatro regioes concatenadas, incluindo
   regioes vazias. Derivar o global pela soma. Guardar contagens sem normalizar.
5. **Metricas**: guardar erro quadratico medio de quantizacao por imagem,
   esparsidade, dimensoes, tempos de encoding e armazenamento.

## Artefactos

```text
data/phase_a/
  descriptors/<assinatura_extrator>/     # um NPZ por imagem
  <assinatura_experiencia>/
    config.json
    manifest.csv                       # IDs, categorias e splits
    inspection.png
    sample/
      descriptors.npy
      images.csv
    vocabularies/
      bovw_K256_C0.npz                  # inclui words, type, num_words, pca_dim
      bovw_K256_C0.json                 # tempo e tamanho da amostra
      ...
    encodings/bovw_K256_C0/
      motorbike_train.npz
      motorbike_train.json
      ...
```

Cada arquivo de encoding contem `file_names`, `histograms` (N x 4K) e
`global_histograms` (N x K). Nunca substituir os originais por vetores normalizados.
Nas fases seguintes, carregar uma das representacoes e comparar nenhuma
normalizacao, L1, L2 e Hellinger, sem repetir extracao ou clustering.

## Protocolo e limites

- A divisao interna esta guardada em `data/splits_internal/manifest.csv`, com
  uma linha por imagem e colunas binarias para todas as categorias. Usa
  estratificacao multilabel iterativa, seed 42: 1978 treino, 494 validacao e
  2474 teste (o val original, intacto). Uma imagem nunca muda de conjunto
  entre categorias. Os TXT derivados permitem reutilizar os loaders existentes.
- Reproduzir com `python -m our_src.make_splits`. A configuracao da fase A
  aponta para `data/splits_internal`. O pipeline atual processa train/val
  internos; a extracao do teste fica para a avaliacao final.
- O vocabulário e ajustado apenas em `train`; a sobreposicao train/val e rejeitada.
- Registar no relatorio hardware, versoes e definicao dos tempos medidos. O tempo
  de encoding exclui extracao/leitura de descritores e escrita dos resultados.
- O cache identifica configuracao, conteudo dos splits e codigo do extrator.
  Pressupoe imagens imutaveis e nomes de imagem unicos. Se substituir imagens ou
  atualizar dependencias, usar um novo `output_dir` para evitar caches antigos.
- Os encoders precomputados fornecidos ficam intactos. Nao misturar os seus
  resultados com os novos sem confirmar o protocolo e pre-processamento.
- O script ainda nao treina SVM nem seleciona K por AP: isso pertence as fases
  seguintes. A inspecao das imagens exige a analise e anotacoes do grupo.
