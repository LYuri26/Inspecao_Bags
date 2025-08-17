import cv2
import os
from pathlib import Path

# Caminhos
PASTA_ENTRADA = Path(
    "/home/lenon/Documentos/GitHub/Inspecao_Sacolas/dataset_sacolas/baixadas"
)
PASTA_SAIDA = Path(
    "/home/lenon/Documentos/GitHub/Inspecao_Sacolas/dataset_sacolas/images/train"
)
RESOLUCAO = (640, 480)
EXTENSAO = ".jpg"

# Cria a pasta de saída se não existir
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

# Percorre imagens na pasta de entrada
for idx, arquivo in enumerate(PASTA_ENTRADA.glob("*")):
    if not arquivo.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
        continue

    img = cv2.imread(str(arquivo))
    if img is None:
        print(f"[ERRO] Falha ao ler {arquivo.name}")
        continue

    img_resized = cv2.resize(img, RESOLUCAO)
    nome_saida = f"img_{idx:04d}{EXTENSAO}"
    caminho_saida = PASTA_SAIDA / nome_saida
    cv2.imwrite(str(caminho_saida), img_resized)
    print(f"[SALVO] {caminho_saida}")
