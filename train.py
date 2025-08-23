import os
import argparse
import shutil
from pathlib import Path
import logging
import sys
import cv2
import random
import numpy as np
from ultralytics import YOLO
import albumentations as A
from collections import defaultdict
from dataclasses import dataclass
import torch

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ---------------- Semente fixa para reprodutibilidade ----------------
def fix_seed(seed: int = 42):
    """Configura semente para todos os componentes aleatórios"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


fix_seed(42)

# ---------------- Configuração padrão ----------------
DEFAULT_CONFIG = {
    "model": "yolov8s.pt",
    "epochs": 200,
    "imgsz": 896,
    "batch": 16,
    "project": "runs/train",
    "name": "detector_sacola_s",
    "yaml_path": "dataset_sacolas/sacolas.yaml",
    "patience": 50,
    "output_model": "modelos/detector_sacola.pt",
    "aug_factor": 10,
    "balance_classes": True,
    "phase_training": True,
    "phase1_epochs": 150,
}


@dataclass
class TrainConfig:
    model: str
    data: Path
    epochs: int
    batch: int
    imgsz: int
    project: Path
    name: str
    device: str
    optimizer: str
    workers: int
    patience: int
    output_model: Path
    aug_factor: int
    balance_classes: bool
    phase_training: bool
    phase1_epochs: int


# ---------------- Funções utilitárias ----------------
def verificar_estrutura(yaml_path: str):
    """Valida a estrutura do dataset YOLO"""
    base_path = os.path.dirname(yaml_path)
    required_dirs = ["images/train", "labels/train", "images/val", "labels/val"]

    missing = []
    empty = []

    for dir_path in required_dirs:
        full_path = os.path.join(base_path, dir_path)
        if not os.path.exists(full_path):
            missing.append(full_path)
        elif not os.listdir(full_path):
            empty.append(full_path)

    if missing or empty:
        error_msg = []
        if missing:
            error_msg.append(f"Diretórios não encontrados: {', '.join(missing)}")
        if empty:
            error_msg.append(f"Diretórios vazios: {', '.join(empty)}")
        raise ValueError("\n".join(error_msg))

    logger.info(f"✅ Estrutura de dados válida em: {base_path}")


def configurar_argumentos():
    """Configura os argumentos de linha de comando"""
    parser = argparse.ArgumentParser(
        description="Treinador YOLOv8s para Detecção de Sacolas"
    )

    for k, v in DEFAULT_CONFIG.items():
        parser.add_argument(
            f"--{k}", type=type(v), default=v, help=f"{k} (padrão: {v})"
        )

    parser.add_argument(
        "--reset", action="store_true", help="Limpar treinamentos anteriores"
    )
    parser.add_argument(
        "--augmentar_dataset",
        action="store_true",
        help="Aplicar data augmentation extra",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Retomar treino de modelo existente"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Dispositivo para treinamento",
    )

    return parser.parse_args()


def limpar_diretorios_anteriores(project_dir):
    """Remove diretórios de treinamentos anteriores"""
    if os.path.exists(project_dir):
        shutil.rmtree(project_dir, ignore_errors=True)
        logger.info("⚠️ Diretórios de treino anteriores removidos")


def preparar_diretorio_saida(output_path):
    """Prepara o diretório de saída"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info(f"✅ Diretório de saída preparado: {os.path.dirname(output_path)}")


def determinar_dispositivo(device_pref):
    """Determina o melhor dispositivo para treinamento"""
    if device_pref == "cpu":
        return "cpu"
    elif device_pref == "cuda" and torch.cuda.is_available():
        return "cuda"
    elif device_pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


# ---------------- Pipeline de Augmentation ----------------
def criar_augmenter():
    """Cria pipeline de augmentação com Albumentations"""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.15, rotate_limit=15, p=0.6
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.25, contrast_limit=0.25, p=0.6
            ),
            A.HueSaturationValue(
                hue_shift_limit=12, sat_shift_limit=18, val_shift_limit=12, p=0.4
            ),
            A.MotionBlur(blur_limit=5, p=0.2),
            A.GaussNoise(var_limit=(10.0, 60.0), p=0.25),
            A.RandomShadow(p=0.1),
            A.RandomSunFlare(p=0.1),
            A.CLAHE(p=0.2),
            A.RandomGamma(p=0.2),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )


def aplicar_augmentations(im_path, lbl_path, out_img_dir, out_lbl_dir, augmenter, idx):
    """Aplica augmentações em uma imagem e seu label"""
    try:
        image = cv2.imread(im_path)
        if image is None:
            raise ValueError(f"Não foi possível ler a imagem: {im_path}")
        if not os.path.exists(lbl_path):
            logger.warning(f"Label não encontrado: {lbl_path}")
            return 0

        # Ler labels
        with open(lbl_path, "r") as f:
            lines = [l.strip().split() for l in f if l.strip()]
            classes = [int(l[0]) for l in lines]
            bboxes = [list(map(float, l[1:])) for l in lines]

        # Aplicar transformações
        transformed = augmenter(image=image, bboxes=bboxes, class_labels=classes)

        # Salvar resultados
        out_img = os.path.join(out_img_dir, f"aug_{idx}_{os.path.basename(im_path)}")
        out_lbl = os.path.join(out_lbl_dir, f"aug_{idx}_{os.path.basename(lbl_path)}")

        cv2.imwrite(out_img, transformed["image"])
        with open(out_lbl, "w") as f:
            for cls, bbox in zip(transformed["class_labels"], transformed["bboxes"]):
                f.write(f"{cls} {' '.join(f'{x:.6f}' for x in bbox)}\n")

        return 1
    except Exception as e:
        logger.error(f"Erro na augmentação {im_path}: {e}")
        return 0


# ---------------- Balanceamento de Dataset ----------------
def coletar_imagens_por_classe(img_dir, lbl_dir):
    """Coleta imagens organizadas por classe"""
    class_imgs = defaultdict(list)
    valid_extensions = (".jpg", ".jpeg", ".png")

    for f in os.listdir(img_dir):
        if not f.lower().endswith(valid_extensions):
            continue

        lbl_file = os.path.join(lbl_dir, f.rsplit(".", 1)[0] + ".txt")
        if not os.path.exists(lbl_file):
            continue

        with open(lbl_file, "r") as lf:
            classes = [int(l.strip().split()[0]) for l in lf if l.strip()]

        for cls in classes:
            class_imgs[cls].append((os.path.join(img_dir, f), lbl_file))

    return class_imgs


def expandir_dataset_balanceado(yaml_path, aug_factor=5):
    """Expande dataset balanceando classes minoritárias"""
    base = os.path.dirname(yaml_path)
    img_dir = os.path.join(base, "images/train")
    lbl_dir = os.path.join(base, "labels/train")
    augmenter = criar_augmenter()
    total_aug = 0

    class_imgs = coletar_imagens_por_classe(img_dir, lbl_dir)
    if not class_imgs:
        raise ValueError("Nenhuma imagem com label encontrada para augmentation")

    # Encontrar a classe com mais exemplos
    max_count = max(len(v) for v in class_imgs.values())

    logger.info("📊 Distribuição de classes antes do balanceamento:")
    for cls, imgs in class_imgs.items():
        logger.info(f"  Classe {cls}: {len(imgs)} imagens")

    # Balancear classes
    for cls, imgs in class_imgs.items():
        needed = max(0, max_count - len(imgs))
        if needed == 0:
            continue

        logger.info(
            f"⚖️  Balanceando classe {cls}: gerando {needed} exemplos adicionais"
        )

        # Repetir lista de imagens para atingir a quantidade necessária
        imgs_to_aug = (imgs * ((needed // len(imgs)) + 1))[:needed]

        for i, (img_path, lbl_path) in enumerate(imgs_to_aug):
            for n in range(aug_factor):
                total_aug += aplicar_augmentations(
                    img_path, lbl_path, img_dir, lbl_dir, augmenter, f"{cls}_{i}_{n}"
                )

    logger.info(f"✅ Dataset expandido balanceado com {total_aug} novas imagens")


def expandir_dataset_automatico(yaml_path, aug_factor=5):
    """Expande dataset com augmentação automática"""
    base = os.path.dirname(yaml_path)
    img_dir = os.path.join(base, "images/train")
    lbl_dir = os.path.join(base, "labels/train")
    augmenter = criar_augmenter()
    total_aug = 0

    imagens = [
        f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not imagens:
        raise ValueError("Nenhuma imagem encontrada para augmentation")

    for idx, img_file in enumerate(imagens):
        img_path = os.path.join(img_dir, img_file)
        lbl_path = os.path.join(lbl_dir, img_file.rsplit(".", 1)[0] + ".txt")

        if not os.path.exists(lbl_path):
            logger.warning(f"Label não encontrado: {lbl_path}")
            continue

        for n in range(aug_factor):
            total_aug += aplicar_augmentations(
                img_path, lbl_path, img_dir, lbl_dir, augmenter, f"{idx}_{n}"
            )

    logger.info(f"✅ Dataset expandido com {total_aug} novas imagens")


# ---------------- Treinamento em Duas Fases ----------------
def treinar_em_fases(cfg: TrainConfig):
    """Treinamento em duas fases para melhor convergência"""
    logger.info(f"🧠 Dispositivo de treino: {cfg.device}")

    # Fase 1: Treinamento inicial sem early stopping
    logger.info(f"🚀 Iniciando Fase 1 (até época {cfg.phase1_epochs}, sem patience)")

    modelo = YOLO(cfg.model)
    resultados_fase1 = modelo.train(
        data=str(cfg.data),
        epochs=cfg.phase1_epochs,
        batch=cfg.batch,
        imgsz=cfg.imgsz,
        project=str(cfg.project),
        name=cfg.name + "_phase1",
        device=cfg.device,
        optimizer=cfg.optimizer,
        workers=cfg.workers,
        patience=0,  # Sem early stopping na fase 1
        cos_lr=True,
        warmup_epochs=5,
        close_mosaic=10,
        multi_scale=True,
        cache=True,
        amp=True,
        lr0=0.01,
        weight_decay=0.0005,
        save_period=10,
    )

    # Encontrar melhor modelo da fase 1
    best_model_phase1 = Path(resultados_fase1.save_dir) / "weights" / "best.pt"
    if not best_model_phase1.exists():
        raise FileNotFoundError("❌ Nenhum best.pt encontrado após a Fase 1")

    # Fase 2: Treinamento refinado com early stopping
    logger.info(
        f"🚀 Iniciando Fase 2 (até época {cfg.epochs}, patience={cfg.patience})"
    )

    modelo_fase2 = YOLO(str(best_model_phase1))
    resultados_fase2 = modelo_fase2.train(
        data=str(cfg.data),
        epochs=cfg.epochs,
        batch=cfg.batch,
        imgsz=cfg.imgsz,
        project=str(cfg.project),
        name=cfg.name + "_phase2",
        device=cfg.device,
        optimizer=cfg.optimizer,
        workers=cfg.workers,
        patience=cfg.patience,
        resume=False,
        cos_lr=True,
        lr0=0.001,  # Learning rate menor na fase 2
        weight_decay=0.0001,
    )

    # Salvar modelo final
    final_model = Path(resultados_fase2.save_dir) / "weights" / "best.pt"
    if final_model.exists():
        shutil.copy(final_model, cfg.output_model)
        logger.info(f"✅ Modelo final salvo em: {cfg.output_model}")

        # Copiar também o último modelo
        last_model = Path(resultados_fase2.save_dir) / "weights" / "last.pt"
        if last_model.exists():
            shutil.copy(last_model, cfg.output_model.parent / "last.pt")
    else:
        logger.error("❌ Modelo final não encontrado!")


def treinamento_direto(cfg: TrainConfig):
    """Treinamento direto (sem fases)"""
    logger.info(f"🧠 Dispositivo de treino: {cfg.device}")

    modelo = YOLO(cfg.model)
    resultados = modelo.train(
        data=str(cfg.data),
        epochs=cfg.epochs,
        batch=cfg.batch,
        imgsz=cfg.imgsz,
        project=str(cfg.project),
        name=cfg.name,
        device=cfg.device,
        optimizer=cfg.optimizer,
        workers=cfg.workers,
        patience=cfg.patience,
        exist_ok=False,
        cos_lr=True,
        warmup_epochs=5,
        close_mosaic=10,
        multi_scale=True,
        cache=True,
        amp=True,
        lr0=0.01,
        weight_decay=0.0005,
        save_period=10,
    )

    # Salvar modelo
    save_dir = Path(resultados.save_dir)
    caminho_modelo = save_dir / "weights" / "best.pt"

    if caminho_modelo.exists():
        shutil.copy(caminho_modelo, cfg.output_model)
        logger.info(f"✅ Modelo salvo: {cfg.output_model}")

        # Copiar também o último modelo
        last_model = save_dir / "weights" / "last.pt"
        if last_model.exists():
            shutil.copy(last_model, cfg.output_model.parent / "last.pt")
    else:
        logger.error(f"❌ Arquivo não encontrado: {caminho_modelo}")


# ---------------- Função Principal ----------------
def main():
    try:
        args = configurar_argumentos()

        # Verificar e preparar caminhos
        yaml_path = os.path.abspath(args.yaml_path)
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Arquivo de dataset não encontrado: {yaml_path}")

        # Determinar dispositivo
        device = determinar_dispositivo(args.device)
        workers = max(2, min(8, (os.cpu_count() or 8) - 1))

        # Configuração de treinamento
        cfg = TrainConfig(
            model=args.model,
            data=Path(yaml_path),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            project=Path(args.project),
            name=args.name,
            device=device,
            optimizer="AdamW",
            workers=workers,
            patience=args.patience,
            output_model=Path(args.output_model),
            aug_factor=args.aug_factor,
            balance_classes=args.balance_classes,
            phase_training=args.phase_training,
            phase1_epochs=args.phase1_epochs,
        )

        # Limpar diretórios anteriores se solicitado
        if args.reset:
            limpar_diretorios_anteriores(str(cfg.project / cfg.name))

        # Preparar diretório de saída
        preparar_diretorio_saida(str(cfg.output_model))

        # Verificar estrutura do dataset
        verificar_estrutura(str(cfg.data))

        # Augmentação de dataset se solicitado
        if args.augmentar_dataset:
            if args.balance_classes:
                expandir_dataset_balanceado(str(cfg.data), args.aug_factor)
            else:
                expandir_dataset_automatico(str(cfg.data), args.aug_factor)

        # Executar treinamento
        if args.phase_training:
            treinar_em_fases(cfg)
        else:
            treinamento_direto(cfg)

        logger.info("✅ Treinamento finalizado com sucesso!")

    except Exception as e:
        logger.error(f"❌ Erro durante o treinamento: {e}")
        import traceback

        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
