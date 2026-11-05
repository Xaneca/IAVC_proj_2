# Executar as etapas encadeadas

Scripts instalados em `C:\IAVC\IAVC_proj_2\code`. No PowerShell:

```powershell
cd C:\IAVC\IAVC_proj_2\code
.\run_part_1.ps1
```

Executa A → B → C → D → E → F → G, sequencialmente, no ambiente
`.venv-stage-a`. Usa as configurações completas existentes em `our_src`.
A calcula automaticamente os centroides em falta com K-means, usando apenas
imagens do treino interno, e reutiliza os existentes compatíveis. Não volta
a gerar os splits. Guarda os centroides em `stageA/centroids/`.
Os resultados ficam nas pastas `results/stageA/` a `results/stageG/`.

Para ver os comandos e validar os caminhos sem iniciar experiências:

```powershell
.\run_part_1.ps1 -DryRun
```

Para retomar a partir de uma etapa (repete essa etapa inteira):

```powershell
.\run_part_1.ps1 -Start E
```

Para executar apenas um intervalo:

```powershell
.\run_part_1.ps1 -Start C -End E
```

As dependências anteriores ao intervalo têm de estar completas. O script
não ignora automaticamente etapas nem retoma dentro de um treino interrompido.
Se uma etapa falhar, a cadeia para imediatamente. Ctrl+C termina o processo
de treino em execução e regista a interrupção. Cada execução guarda logs por
etapa e estado global em `pipeline_logs/<data_hora>/pipeline.json`.

Smoke de integração A–G com os dados reais pequenos e vocabulário K=32:

```powershell
.\run_part_1.ps1 -Smoke -PerClass 2
```

O smoke usa pastas `results/stageA/smoke` a `results/stageG/smoke`; G usa 50%/100% do treino
pequeno com dez seeds. Não altera os defaults da experiência completa.
Os centroides smoke são calculados apenas com o treino pequeno e guardados em
`results/stageA/smoke/centroids/`, separados dos centroides da experiência completa.
`-Start`/`-End` aplicam-se apenas ao modo completo; smoke corre sempre A–G.

Também pode ser executado diretamente:

```powershell
.\.venv-stage-a\Scripts\python.exe .\run_part_1.py --dry-run
```
