# Pipeline de flanges: v7 × v8

Nov 11, 2026 · @ba

## Resumo

A v8 faz treino e inferência usarem exatamente os mesmos dados de entrada e corrige um bug de rótulo na augmentation. A arquitetura, a loss e os hiperparâmetros são os mesmos da v7.

Na v7 havia três desalinhamentos. O modelo usado na inferência (`best_v7.pth`, treinado com augmentation pesada) não era o modelo das métricas do artigo (`model_no_aug.pth`, da ablação). A inferência passava zeros em RGB e intensidade, usava voxel de 1 cm e outra rotina de curvatura. E a métrica publicada foi medida em 400 chunks com argmax, não na cena inteira.

A v8 ainda não rodou nos dados reais. Este relatório compara código e desenho, não resultados.

## Diferenças por etapa

| Etapa | Item | v7 | v8 |
| --- | --- | --- | --- |
| Geral | Organização | Cada script com sua cópia de leitura, voxel e features | `common_v8.py` único, importado pelos três scripts |
| Pré-processamento | Curvatura | sklearn `query_radius` + laço em Python, usando os primeiros 30 índices do raio | Open3D `estimate_covariances` (raio 5 cm, até 30 vizinhos), igual à inferência |
| Pré-processamento | Metadados | `was_mm` | `was_mm`, `has_rgb`, `has_intensity` |
| Treino | Augmentation | Pesada, sempre ligada | Parâmetro `--aug none\|light\|heavy`; padrão `none`, como no artigo |
| Treino | Features por ponto | 8 (normal, curvatura, RGB, intensidade) | 10 (as 8 + \|n\_z\| + z relativo), como no artigo |
| Treino | Canais de entrada | 11 | 13 |
| Treino | Validação | Âncora com semente, subamostra sem semente | Âncora e subamostra com semente |
| Treino | Teste | Separado, nunca avaliado | Avaliado no fim (400 chunks), salvo em JSON |
| Treino | Semente | Fixa em 42 no código | `--seed`, para repetir com três sementes |
| Treino | Checkpoint | Só os pesos | Pesos + configuração + split dos arquivos |
| Treino | Vizinhos | `argsort` da matriz inteira | `topk` (mesmo resultado, menos memória) |
| Inferência | Modelo | `best_v7.pth` (pesada) | O checkpoint escolhido em `--model` |
| Inferência | RGB e intensidade | Sempre zeros | Valores reais do PLY; zeros só se o arquivo não tiver |
| Inferência | Voxel | 1 cm (Open3D) | 5 mm, mesma função do treino |
| Inferência | Volta aos pontos originais | Vizinho mais próximo por KDTree | Índice `inv` do próprio voxel (exato) |
| Inferência | Limiar | Adaptativo: máx(0,35; 0,7 × p\_máx) | Fixo em 0,5; adaptativo opcional |
| Inferência | Pontos sem cobertura | Viravam equipamento | Herdam os logits do vizinho coberto mais próximo |
| Avaliação | Cena inteira | Só um teste de sanidade em um arquivo | `--eval valid\|test` com métricas agregadas e varredura de limiar |

## Bug de rótulo na augmentation pesada

Na v7, o dropout de pontos e a oclusão de octante copiavam a geometria de um ponto mantido para a posição de um ponto descartado, mas o rótulo do ponto descartado ficava. Um ponto de tubo podia sair rotulado como flange, e o inverso.

O dropout atingia de 0% a 50% dos pontos do chunk, e a oclusão de 5% a 40% em 30% dos chunks. Na oclusão havia um segundo erro: coordenadas e normais eram copiadas de pontos diferentes.

Isso afeta a leitura da ablação publicada. A configuração Heavy teve IoU de flange de 38,98% contra 51,62% da No-Aug, com recall maior e precisão menor. Parte dessa queda pode vir do ruído de rótulo, não da augmentation em si. A configuração Light não usa essas duas transformações e não é afetada.

Na v8 a linha inteira é copiada junto com o rótulo.

## O que não mudou

- Arquitetura: PointNet++ com 4 blocos SA, 4 blocos FP, GroupNorm de 16 grupos e dropout de 0,5 na cabeça.
- Loss: Focal (γ = 2, peso 8 para flange) somada à Lovász-Softmax.
- Otimização: AdamW, taxa 1e-3, weight decay 1e-5, cosseno até 1e-6, clipping em 1,0, paciência de 30 épocas.
- Amostragem: chunks de 8.192 pontos, raios de 0,5, 1,0 e 2,0 m, 50% das âncoras em flange.
- Split por equipamento: 19 treino, 4 validação, 5 teste com a semente 42.
- Inferência: grade de âncoras a cada 0,4 m, raio de 1,0 m, média dos logits.

## Como comparar os dois nos dados reais

A comparação justa usa a mesma métrica de cena inteira nos mesmos 5 equipamentos de teste.

1. Rodar `python preprocess_v8.py` para gerar `DATA_PREPROCESSED_v8`.
2. Treinar a v8 sem augmentation: `python train_v8.py --aug none --seed 42`.
3. Escolher o limiar na validação: `python inference_v8.py --eval valid`.
4. Medir no teste com o limiar escolhido: `python inference_v8.py --eval test --thresh <valor>`.
5. Medir a v7 do mesmo jeito: rodar a inferência v7 com `best_v7.pth` nos 5 equipamentos de teste e comparar com os rótulos.
6. Refazer a ablação com o bug corrigido: `--aug light` e `--aug heavy`, mesma semente.
7. Repetir os passos 2 a 4 com mais duas sementes e reportar média e desvio-padrão.

O passo 5 exige um script pequeno, porque a inferência v7 não calcula métricas contra rótulos em lote. O `run_sanity_check` da v7 faz isso para um arquivo e serve de base.

## Limites e o que ainda não foi verificado

- A v8 foi testada de ponta a ponta só em nuvens sintéticas pequenas, em CPU, com 1.024 pontos por chunk. Falta rodar nos 28 equipamentos, na GPU, com 8.192 pontos.
- Não há número de ganho. Nenhuma métrica da v8 nos dados reais existe ainda.
- Os checkpoints da v7 não carregam na v8, porque a entrada passou de 11 para 13 canais.
- O voxel de 5 mm deixa a inferência mais lenta que a v7 com 1 cm. Para acelerar, mudar `VOXEL_SIZE_M` em `common_v8.py` e reprocessar, para treino e inferência continuarem iguais.
- Que a Figura 8 do artigo vem do treino com augmentation pesada é uma inferência a partir das curvas (150 épocas, validação melhor que treino). Confirmar no `history_v7.csv`.
- Se as nuvens de produção não tiverem cor nem intensidade, um modelo treinado com esses canais reais recebe zeros que nunca viu. Nesse caso vale treinar só com geometria ou usar `--aug light`, que zera RGB em 30% dos chunks.
