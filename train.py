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
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------- Semente fixa ----------------
def fix_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


fix_seed(42)

# ---------------- Config padrão ----------------
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


# ---------------- Funções utilitárias ----------------
def verificar_estrutura(yaml_path: str):
    base_path = os.path.dirname(yaml_path)
    required_dirs = ["images/train", "labels/train", "images/val", "labels/val"]

    for dir_path in required_dirs:
        full_path = os.path.join(base_path, dir_path)
        if not os.path.exists(full_path):
            raise ValueError(f"Diretório não encontrado: {full_path}")
        if not os.listdir(full_path):
            raise ValueError(f"Diretório vazio: {full_path}")

    logger.info(f"✅ Estrutura de dados válida em: {base_path}")


def configurar_argumentos():
    parser = argparse.ArgumentParser(
        description="Treinador YOLOv8 (Small) para Sacolas"
    )
    for k, v in DEFAULT_CONFIG.items():
        parser.add_argument(f"--{k}", type=type(v), default=v)
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
    return parser.parse_args()


def limpar_diretorios_anteriores(project_dir):
    if os.path.exists(project_dir):
        shutil.rmtree(project_dir, ignore_errors=True)
        logger.info("⚠️ Diretórios de treino anteriores removidos")


def preparar_diretorio_saida(output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)


# ---------------- Treinamento em duas fases ----------------
def treinar_em_fases(cfg: TrainConfig, patience_start_epoch=150):
    logger.info(f"🧠 Dispositivo de treino: {cfg.device}")

    modelo = YOLO(cfg.model)

    # ----- Fase 1 (até patience_start_epoch, sem early stopping)
    logger.info(f"🚀 Iniciando Fase 1 (até época {patience_start_epoch}, sem patience)")
    resultados = modelo.train(
        data=str(cfg.data),
        epochs=patience_start_epoch,
        batch=cfg.batch,
        imgsz=cfg.imgsz,
        project=str(cfg.project),
        name=cfg.name + "_phase1",
        device=cfg.device,
        optimizer=cfg.optimizer,
        workers=cfg.workers,
        patience=0,
        cos_lr=True,
        warmup_epochs=5,
        close_mosaic=20,
        multi_scale=True,
        cache=True,
        amp=True,
        lr0=0.002,
        weight_decay=0.0005,
    )

    best_model = Path(resultados.save_dir) / "weights" / "best.pt"
    if not best_model.exists():
        raise FileNotFoundError("❌ Nenhum best.pt encontrado após a Fase 1")

    # ----- Fase 2 (continua até epochs, com patience ativo)
    logger.info(
        f"🚀 Iniciando Fase 2 (até época {cfg.epochs}, patience={cfg.patience})"
    )
    modelo = YOLO(str(best_model))
    resultados = modelo.train(
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
    )

    final_model = Path(resultados.save_dir) / "weights" / "best.pt"
    if final_model.exists():
        shutil.copy(final_model, cfg.output_model)
        logger.info(f"✅ Modelo final salvo em: {cfg.output_model}")
    else:
        logger.error("❌ Modelo final não encontrado!")


# ---------------- Augmentation pipeline ----------------
def criar_augmenter():
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
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )


def aplicar_augmentations(im_path, lbl_path, out_img_dir, out_lbl_dir, augmenter, idx):
    try:
        image = cv2.imread(im_path)
        if image is None:
            raise ValueError(f"Não foi possível ler a imagem: {im_path}")
        if not os.path.exists(lbl_path):
            logger.warning(f"Label não encontrado: {lbl_path}")
            return 0

        with open(lbl_path, "r") as f:
            lines = [l.strip().split() for l in f if l.strip()]
            classes = [int(l[0]) for l in lines]
            bboxes = [list(map(float, l[1:])) for l in lines]

        transformed = augmenter(image=image, bboxes=bboxes, class_labels=classes)

        out_img = os.path.join(out_img_dir, f"aug_{idx}_{os.path.basename(im_path)}")
        out_lbl = os.path.join(out_lbl_dir, f"aug_{idx}_{os.path.basename(lbl_path)}")
        cv2.imwrite(out_img, transformed["image"])
        with open(out_lbl, "w") as f:
            for cls, bbox in zip(transformed["class_labels"], transformed["bboxes"]):
                f.write(f"{cls} {' '.join(f'{x:.6f}' for x in bbox)}\n")
        return 1
    except Exception as e:
        logger.error(f"Erro augment {im_path}: {e}")
        return 0


# ---------------- Dataset balanceado ----------------
def coletar_imagens_por_classe(img_dir, lbl_dir):
    class_imgs = defaultdict(list)
    for f in os.listdir(img_dir):
        if not f.lower().endswith((".jpg", ".jpeg", ".png")):
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
    base = os.path.dirname(yaml_path)
    img_dir = os.path.join(base, "images/train")
    lbl_dir = os.path.join(base, "labels/train")
    augmenter = criar_augmenter()
    total_aug = 0

    class_imgs = coletar_imagens_por_classe(img_dir, lbl_dir)
    if not class_imgs:
        raise ValueError("Nenhuma imagem com label encontrada para augmentation")

    max_count = max(len(v) for v in class_imgs.values())

    for cls, imgs in class_imgs.items():
        needed = max_count - len(imgs)
        imgs_to_aug = imgs * ((needed // len(imgs)) + 1) if imgs else []
        for i, (img_path, lbl_path) in enumerate(imgs_to_aug[:needed]):
            for n in range(aug_factor):
                total_aug += aplicar_augmentations(
                    img_path, lbl_path, img_dir, lbl_dir, augmenter, f"{cls}_{i}_{n}"
                )

    logger.info(f"✅ Dataset expandido balanceado com {total_aug} novas imagens")


def expandir_dataset_automatico(yaml_path, aug_factor=5):
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


# ---------------- Treinamento ----------------
def treinar_modelo(cfg: TrainConfig):
    try:
        verificar_estrutura(str(cfg.data))
        preparar_diretorio_saida(str(cfg.output_model))

        modelo = YOLO(cfg.model)

        overrides = {
            "data": str(cfg.data),
            "epochs": cfg.epochs,
            "batch": cfg.batch,
            "imgsz": cfg.imgsz,
            "project": str(cfg.project),
            "name": cfg.name,
            "device": cfg.device,
            "optimizer": cfg.optimizer,
            "workers": cfg.workers,
            "patience": cfg.patience,
            "exist_ok": False,
            # Ajustes de qualidade
            "cos_lr": True,
            "warmup_epochs": 5,
            "close_mosaic": 20,
            "multi_scale": True,
            "cache": True,
            "amp": True,
            "lr0": 0.002,
            "weight_decay": 0.0005,
            # Augmentações nativas YOLO
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 5.0,
            "translate": 0.1,
            "scale": 0.7,
            "shear": 0.0,
            "flipud": 0.0,
            "fliplr": 0.5,
            "mosaic": 1.0,
            "mixup": 0.15,
            "copy_paste": 0.15,
        }

        logger.info(f"🧠 Dispositivo de treino: {cfg.device}")
        resultados = modelo.train(**overrides)

        save_dir = Path(resultados.save_dir)
        caminho_modelo = save_dir / "weights" / "best.pt"

        if caminho_modelo.exists():
            shutil.copy(caminho_modelo, cfg.output_model)
            logger.info(f"✅ Modelo salvo: {cfg.output_model}")
        else:
            logger.error(f"❌ Arquivo não encontrado: {caminho_modelo}")

    except Exception as e:
        logger.error(f"Erro treinamento: {e}")
        raise


# ---------------- Main ----------------
if __name__ == "__main__":
    try:
        yaml_path = os.path.abspath(os.path.join("dataset_sacolas", "sacolas.yaml"))
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Arquivo de dataset não encontrado: {yaml_path}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        workers = max(2, min(8, (os.cpu_count() or 8) - 1))

        cfg = TrainConfig(
            model=DEFAULT_CONFIG["model"],
            data=Path(yaml_path),
            epochs=DEFAULT_CONFIG["epochs"],
            batch=DEFAULT_CONFIG["batch"],
            imgsz=DEFAULT_CONFIG["imgsz"],
            project=Path(DEFAULT_CONFIG["project"]),
            name=DEFAULT_CONFIG["name"],
            device="cpu",  # <---- força CPU
            optimizer="AdamW",
            workers=workers,
            patience=DEFAULT_CONFIG["patience"],
            output_model=Path(DEFAULT_CONFIG["output_model"]),
        )
        preparar_diretorio_saida(str(cfg.output_model))
        treinar_em_fases(cfg, patience_start_epoch=150)

    except Exception as e:
        logger.error(f"Erro: {e}")
        sys.exit(1)
