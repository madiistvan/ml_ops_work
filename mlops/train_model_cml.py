import hydra
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data_utils
from lightning.lightning_nn import LightningNeuralNet
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from models.model import NeuralNet


def load_dataset(batch_size, path="data/processed"):
    train_images = torch.load(f"{path}/train_images.pt")
    train_targets = torch.load(f"{path}/train_targets.pt")
    train = data_utils.TensorDataset(train_images, train_targets)
    trainloader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)

    test_images = torch.load(f"{path}/test_images.pt")
    test_targets = torch.load(f"{path}/test_targets.pt")
    test = data_utils.TensorDataset(test_images, test_targets)
    testloader = data_utils.DataLoader(test, batch_size=batch_size, shuffle=False)

    return trainloader, testloader


def save_loss_plot(train_losses, test_losses):
    plt.plot(train_losses, label="Training loss")
    plt.plot(test_losses, label="Test loss")
    plt.legend()
    plt.title("Losses")
    plt.savefig("reports/figures/loss.png")
    print("Loss plot saved.")


def save_model(model, epoch):
    torch.save(model.state_dict(), f"mlops/checkpoints/checkpoint_{epoch}.pth")
    print("Model saved.")


def get_model(cfg):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_hparams = cfg.hyperparameters
    model = NeuralNet(
        c1out=model_hparams["c1out"],
        c2out=model_hparams["c2out"],
        c3out=model_hparams["c3out"],
        fc1out=model_hparams["fc1out"],
        fc2out=model_hparams["fc2out"],
        fc3out=model_hparams["fc3out"],
        p_drop=model_hparams["p_drop"],
    )
    model.to(DEVICE)

    return model


def train(train_cfg, model_cfg):
    """
    Trains the model and saves the trained model to mlops/checkpoints/checkpoint_[epoch].pth and saves the loss plot to reports/figures/loss.png
    """

    train_hparams = train_cfg.hyperparameters

    model = get_model(model_cfg)
    lightning_model = LightningNeuralNet(model, train_hparams["lr"])

    trainloader, testloader = load_dataset(train_hparams["batch_size"])
    checkpoint_callback = ModelCheckpoint(dirpath="mlops/checkpoints", monitor="Validation loss", mode="min")

    trainer = Trainer(
        accelerator="gpu",
        max_epochs=train_hparams["epochs"],
        callbacks=[checkpoint_callback],
        precision="16",
        logger=WandbLogger(project="mlops"),
    )
    trainer.fit(model=lightning_model, train_dataloaders=trainloader, val_dataloaders=testloader)

    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

    preds, target = [], []
    for batch in trainloader:
        x, y = batch
        probs = model(x)
        preds.append(probs.argmax(dim=-1))
        target.append(y.detach())

    target = torch.cat(target, dim=0)
    preds = torch.cat(preds, dim=0)

    report = classification_report(target, preds)
    with open("classification_report.txt", "w") as outfile:
        outfile.write(report)
    confmat = confusion_matrix(target, preds)
    disp = ConfusionMatrixDisplay(
        cm=confmat,
    )
    plt.savefig("confusion_matrix.png")


def main():
    hydra.initialize(config_path="conf")
    train_cfg = hydra.compose("train_conf.yaml")
    model_cfg = hydra.compose("model_conf.yaml")
    train(train_cfg, model_cfg)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error during training.")
        print(e)
