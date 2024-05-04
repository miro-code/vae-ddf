import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pytorch_lightning import Trainer
from argparse import ArgumentParser
from models import ConvVAE
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch import Trainer, seed_everything
from datasets import MNISTDebugDataModule as MNISTDataModule #TODO CHANGE
import os

if __name__ == "__main__":
    parser = ArgumentParser(prog="VAE for DDF", description="Train VAE for downstream use with differential decision forests")
    parser.add_argument("--batch_size", type=int, default=512, help="batch size")
    parser.add_argument("--epochs", type=int, default=50, help="num epochs")
    parser.add_argument("--latent-dim", type=int, default=8, help="size of latent dim for our vae")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--kl-coeff", type=int, default=5, help="kl coeff aka beta term in the elbo loss function")
    parser.add_argument("--output_dir", type=str, default=os.path.join("results", 'vae'), help="output directory")
    parser.add_argument("--anomaly_detect", help="Detect anomalies", action="store_true", default=False)
    parser.add_argument("--name", type=str, default="vae-for-ddf", help="wandb name of the run")
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint to load")
    args = parser.parse_args()
    seed_everything(42)
    os.makedirs(args.output_dir, exist_ok=True)
    wandb_logger = WandbLogger(
    name=args.name,
    project=parser.prog,
    save_dir=args.output,
    log_model=True,  # Log checkpoint only at the end of training (to stop my wandb running out of storage!)
    )
    #args without name and output_dir
    config = vars(args)
    config.pop("name")
    config.pop("output_dir")
    wandb_logger.experiment.config.update(config)
    
    model_params = {
        "input_shape" : (1, 28, 28),
        "encoder_conv_filters" : [28, 64, 64],
        "decoder_conv_t_filters" : [64, 28, 1],
        "latent_dim" : args.latent_dim,
        "kl_coeff" : args.kl_coeff,
        "lr" : args.lr,
    }
    val_checkpoint = ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    auto_insert_metric_name=True,
    )

    latest_checkpoint = ModelCheckpoint(
        filename="latest-checkpoint",
        every_n_epochs=1,
        save_top_k=1,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConvVAE(**model_params)
    model.to(device)

    trainer = Trainer(
        accelerator=str(device),
        logger=wandb_logger,
        callbacks=[latest_checkpoint, val_checkpoint],
        detect_anomaly=args.anomaly_detect,
        max_epochs=args.epochs,
    )
    datamodule = MNISTDataModule(batch_size=args.batch_size)
    trainer.fit(
        model, 
        datamodule=datamodule,
        ckpt_path=args.checkpoint,
        )
    trainer.test(datamodule=datamodule)
    wandb_logger.save()
    wandb_logger.finalize()

