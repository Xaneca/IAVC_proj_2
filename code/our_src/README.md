# Stage A: guardar apenas centroides

Politica atual: SIFT, amostra de treino e histogramas nao sao guardados em disco.
So ficam os centroides de cada K-means e a configuracao necessaria para os identificar.
Os splits existentes sao preservados. Para os reproduzir: `python -m our_src.make_splits`.

```powershell
# Executar a partir de code/
.\.venv-stage-a\Scripts\python.exe -m our_src.stage_a
```

`--stage vocabularies` e `--stage all` executam apenas o ajuste dos vocabularios.
Nao iniciam encoding, metricas nem graficos. A configuracao e `our_src/stage_a.json`:
seed 42; K=32,64,128,256,512,1024; todas as imagens do treino interno e ate 400 SIFT por imagem.

Reutiliza `loadSamplesTable` para selecionar imagens, `extractFeatures(cache_dir=None)`
para extrair SIFT sem cache e `train_encoder` para ajustar cada K-means sem PCA.
Cada imagem selecionada e extraida uma vez; uma amostra limitada fica em RAM
e serve todos os K em falta. Apenas treino interno participa no ajuste.

Os unicos novos artefactos da pipeline sao:

```text
data/stage_a_json/models/kmeans/K<K>_seed42_<config>.json
```

Cada ficheiro contem centroides e metadados de origem/configuracao; nao contem
SIFT, amostra, atribuicoes ou histogramas. Modelos existentes compativeis sao
reutilizados. Se faltar outro K numa execucao futura, e necessario extrair
novamente a amostra (deterministica); nao e necessario reajustar os K existentes.

Os ficheiros da execucao antiga interrompida nao sao apagados automaticamente.
Os modulos `a_metrics.py` mantem funcoes disponiveis para futuras analises em RAM,
mas nao sao executados pelo comando atual.

Testes: `python -m our_src.test_stage_a`.
