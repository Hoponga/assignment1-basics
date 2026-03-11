from cs336_basics.optim import *
from cs336_basics.model import *
from cs336_basics.tokenizer import *
from cs336_basics.dataloader import *
from cs336_basics.bpe import *
import numpy as np
import yaml
import wandb


def run_train(model, train_config, metrics): 
    model.to(device)

    # Optional: watch gradients/params
    wandb.watch(model, criterion, log="all", log_freq=100)

    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(config.num_epochs):
        # -------------------
        # Training
        # -------------------
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            # assumes batch = (inputs, targets)
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            # metrics
            train_loss_sum += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)

            # step-level logging
            wandb.log(
                {
                    "train/loss_step": loss.item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "global_step": global_step,
                },
                step=global_step,
            )

            global_step += 1

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        # -------------------
        # Validation
        # -------------------
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss_sum += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total
        epoch_time = time.time() - epoch_start

        # epoch-level logging
        wandb.log(
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "epoch_time_sec": epoch_time,
            },
            step=global_step,
        )

        print(
            f"Epoch {epoch+1}/{config.num_epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = "best_model.pt"
            torch.save(model.state_dict(), ckpt_path)

            # upload checkpoint to W&B
            artifact = wandb.Artifact("best-model", type="model")
            artifact.add_file(ckpt_path)
            wandb.log_artifact(artifact)

    wandb.finish()





if __name__ == "__main__": 
    with open("train_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    train_data_file = config["data"]["train_file"]
    val_data_file = config["data"]["val_file"]

    context_length = config["data"]["context_length"]
    special_tokens = config["data"]["special_tokens"]
    batch_size  = config["training"]["batch_size"]
    device = config["training"]["device"]

    # checkpointing

    checkpoint_save_every = config["checkpoint"]["save_every"]
    checkpoint_dir = config["checkpoint"]["out_dir"]
    checkpoint_src = config["checkpoint"]["resume_from"]


    # Expect pre-tokenized .npy files (run create_dataset.py first)
    vocab_size = config["model"]["vocab_size"]
    dtype = np.uint16 if vocab_size <= 65535 else np.uint32

    train_dataset = load_dataset_mmap(train_data_file, dtype=dtype)
    val_dataset   = load_dataset_mmap(val_data_file,   dtype=dtype)

    model = init_model_from_config(config["model"], checkpoint_src)

    optimizer = AdamW(model.parameters())

    project_name = config["wandb"]["project"]
    run_name = config["wandb"]["run_name"]

    wandb.init(project=project_name, name=run_name, config=config)
    metrics = dict()

    # here our dataloader is just a memory-mapped numpy array
    run_train(model, train_dataset, config, metrics)




    



    



