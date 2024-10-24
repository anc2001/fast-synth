import argparse
import torch
from torch.utils.data import random_split
import json
import numpy as np
import torch.optim as optim
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
import sys
import utils
from tqdm import tqdm
from models.resnet import resnet18

from threedf_dataset import ThreedfDataset, get_categories_list

# -------------------------- -------------------------------------------------------------
class NextCategory(nn.Module):

    def __init__(self, n_input_channels, n_categories, bottleneck_size):
        super(NextCategory, self).__init__()

        activation = nn.LeakyReLU()

        self.cat_prior_img = nn.Sequential(
            resnet18(num_input_channels=n_input_channels, num_classes=bottleneck_size),
            nn.BatchNorm1d(bottleneck_size),
            activation,
        )
        self.cat_prior_counts = nn.Sequential(
            nn.Linear(n_categories, bottleneck_size),
            nn.BatchNorm1d(bottleneck_size),
            activation,
            nn.Linear(bottleneck_size, bottleneck_size),
            nn.BatchNorm1d(bottleneck_size),
            activation,
        )
        self.cat_prior_final = nn.Sequential(
            nn.Linear(2 * bottleneck_size, bottleneck_size),
            nn.BatchNorm1d(bottleneck_size),
            activation,
            # +1 -> the 'stop' category
            nn.Linear(bottleneck_size, n_categories + 1),
        )

    def forward(self, input_scene, catcount):
        cat_img = self.cat_prior_img(input_scene)
        cat_count = self.cat_prior_counts(catcount)
        catlogits = self.cat_prior_final(torch.cat([cat_img, cat_count], dim=-1))
        return catlogits

    def sample(self, input_scene, catcount, temp=1, return_logits=False):
        logits = self.forward(input_scene, catcount)
        if temp != 1.0:
            logits = logits * temp
        # logits[:,-1] += 1
        cat = torch.distributions.Categorical(logits=logits).sample()
        if return_logits:
            return cat, logits
        else:
            return cat


# ---------------------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Module that predicts the category of the next object
    """

    parser = argparse.ArgumentParser(description="cat")
    parser.add_argument("--num-workers", type=int, default=6, metavar="N")
    parser.add_argument("--last-epoch", type=int, default=-1, metavar="N")
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--train-size", type=int, default=5000, metavar="N")
    parser.add_argument("--save-dir", type=str, default="cat_test", metavar="S")
    parser.add_argument("--save-every-n-epochs", type=int, default=5, metavar="N")
    parser.add_argument("--lr", type=float, default=0.0005, metavar="N")

    # This branch specific
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epoch-size", type=int, default=10000)
    parser.add_argument("--latent-dim", type=int, default=200)
    parser.add_argument("--grid-size", type=int, default=256)

    parser.add_argument("--room-type", type=str, required=True)
    parser.add_argument("--bounds-file", type=str, required=True)
    parser.add_argument("--input-dir", type=str, required=True)
    args = parser.parse_args()

    batch_size = args.batch_size
    epoch_size = args.epoch_size 
    latent_dim = args.latent_dim
    grid_size = args.grid_size

    save_dir = args.save_dir
    save_every = args.save_every_n_epochs
    utils.ensuredir(save_dir)

    logfile = open(f"{save_dir}/log.txt", "w")

    def LOG(msg):
        print(msg)
        logfile.write(msg + "\n")
        logfile.flush()

    num_categories = len(get_categories_list(args.room_type))
    num_input_channels = num_categories + 6

    LOG(
        f"Building model... {num_input_channels} input channels, {num_categories} categories, {latent_dim} latent dim"
    )

    model = NextCategory(num_input_channels, num_categories, latent_dim)
    if torch.cuda.is_available():
        model = model.cuda()

    LOG("Building datasets...")

    cat_dataset = ThreedfDataset(
        args.input_dir, "cat", args.room_type, args.bounds_file, args.grid_size
    )

    # Define the sizes of your splits. For example, 80% train, 20% validation
    total_size = len(cat_dataset)
    train_size = int(0.8 * total_size)
    validation_size = total_size - train_size

    # Split the dataset
    train_dataset, validation_dataset = random_split(
        cat_dataset, [train_size, validation_size]
    )

    LOG("Building data loader...")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=args.num_workers, shuffle=True
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=batch_size, num_workers=0, shuffle=True
    )

    optimizer = optim.Adam(list(model.parameters()), lr=args.lr)

    if args.last_epoch < 0:
        load = False
        starting_epoch = 0
    else:
        load = True
        last_epoch = args.last_epoch

    if load:
        LOG("Loading saved models...")
        model.load_state_dict(torch.load(f"{save_dir}/nextcat_{last_epoch}.pt"))
        optimizer.load_state_dict(torch.load(f"{save_dir}/nextcat_optim_backup.pt"))
        starting_epoch = last_epoch + 1

    current_epoch = starting_epoch
    num_seen = 0

    model.train()
    LOG(
        f"=========================== Epoch {current_epoch} ==========================="
    )

    loss_running_avg = 0

#    from utils import save_input_img_as_png
#    for input_img, _, _ in tqdm(train_loader):
#        for i in range(input_img.shape[0]):
#            save_input_img_as_png(input_img, i, save_path=f"{args.save_dir}/{i}.png")
#
    def train():
        global num_seen, current_epoch, loss_running_avg

        for batch_idx, (input_img, t_cat, catcount) in enumerate(train_loader):

            # Get rid of singleton dimesion in t_cat (NLLLoss complains about this)
            t_cat = torch.squeeze(t_cat)

            if torch.cuda.is_available():
                input_img, t_cat, catcount = (
                    input_img.cuda(),
                    t_cat.cuda(),
                    catcount.cuda(),
                )

            optimizer.zero_grad()
            logits = model(input_img, catcount)

            loss = F.cross_entropy(logits, t_cat)
            loss.backward()
            optimizer.step()
            loss_precision = 4
            LOG(f"Loss: {loss.cpu().data.numpy():{loss_precision}.{loss_precision}}")

            num_seen += batch_size

            if num_seen % 800 == 0:
                LOG(f"Examples {num_seen}/{epoch_size}")
            if num_seen > 0 and num_seen % epoch_size == 0:
                validate()
                num_seen = 0
                if current_epoch % save_every == 0:
                    torch.save(
                        model.state_dict(), f"{save_dir}/nextcat_{current_epoch}.pt"
                    )
                    torch.save(
                        optimizer.state_dict(), f"{save_dir}/nextcat_optim_backup.pt"
                    )
                if current_epoch == args.num_epochs:
                    print("training complete")
                    exit()

                current_epoch += 1
                LOG(
                    f"=========================== Epoch {current_epoch} ==========================="
                )

    def validate():
        LOG("Validating")
        model.eval()
        total_loss = 0
        num_correct_top1 = 0
        num_correct_top5 = 0
        num_batches = 0
        num_items = 0

        for batch_idx, (input_img, t_cat, catcount) in enumerate(validation_loader):

            # Get rid of singleton dimesion in t_cat (NLLLoss complains about this)
            t_cat = torch.squeeze(t_cat)

            # uncomment if graphics card
            if torch.cuda.is_available():
                input_img, t_cat, catcount = (
                    input_img.cuda(),
                    t_cat.cuda(),
                    catcount.cuda(),
                )

            with torch.no_grad():
                logits = model(input_img, catcount)
                loss = F.cross_entropy(logits, t_cat)

            total_loss += loss.cpu().data.numpy()

            # TODO: Why is top1 always giving me zero...?
            _, argmax = logits.max(dim=-1)
            num_correct_top1 += (argmax == t_cat).sum()

            lsorted, lsorted_idx = torch.sort(logits, dim=-1, descending=True)
            lsorted_idx_top5 = lsorted_idx[:, 0:5]
            num_correct = torch.zeros(input_img.shape[0])
            if torch.cuda.is_available():
                num_correct = num_correct.cuda()
            for i in range(0, 5):
                correct = lsorted_idx_top5[0:, i] == t_cat
                num_correct = torch.max(num_correct, correct.float())
            num_correct_top5 += num_correct.sum()

            num_batches += 1
            num_items += input_img.shape[0]

        LOG(f"Average Loss: {total_loss / num_batches}")
        LOG(f"Top 1 Accuracy: {num_correct_top1 / num_items}")
        LOG(f"Top 5 Accuracy: {num_correct_top5 / num_items}")
        model.train()

    while True:
        train()
