import torch
import torch.nn as nn

class Generator(nn.Module):
    """Simple Generator for MNIST (28x28) using a fully connected network.
    Takes a noise vector of size `latent_dim` and outputs a 1x28x28 image.
    """
    def __init__(self, latent_dim: int = 100):
        super().__init__()
        self.latent_dim = latent_dim
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 28 * 28),
            nn.Tanh()  # output in [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        img = self.model(z)
        img = img.view(z.size(0), 1, 28, 28)
        return img

class Discriminator(nn.Module):
    """Simple Discriminator for MNIST.
    Takes a 1x28x28 image and outputs a probability.
    """
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.model(img)

class GANTrainer:
    """Encapsulates the training loop for the GAN.

    Args:
        generator: Generator model.
        discriminator: Discriminator model.
        device: torch device (cpu or cuda).
        latent_dim: Dimensionality of the noise vector.
    """
    def __init__(self, generator: Generator, discriminator: Discriminator, device: torch.device, latent_dim: int = 100, lr: float = 2e-4):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device
        self.latent_dim = latent_dim
        self.criterion = nn.BCELoss()
        self.opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    def train(self, dataloader, epochs: int = 20, log_interval: int = 100):
        self.generator.train()
        self.discriminator.train()
        history = {"g_loss": [], "d_loss": []}
        for epoch in range(1, epochs + 1):
            for i, (real_imgs, _) in enumerate(dataloader):
                batch_size = real_imgs.size(0)
                real_imgs = real_imgs.to(self.device)
                # Labels
                valid = torch.ones(batch_size, 1, device=self.device)
                fake = torch.zeros(batch_size, 1, device=self.device)
                # -----------------
                #  Train Generator
                # -----------------
                self.opt_g.zero_grad()
                z = torch.randn(batch_size, self.latent_dim, device=self.device)
                gen_imgs = self.generator(z)
                g_loss = self.criterion(self.discriminator(gen_imgs), valid)
                g_loss.backward()
                self.opt_g.step()
                # ---------------------
                #  Train Discriminator
                # ---------------------
                self.opt_d.zero_grad()
                real_loss = self.criterion(self.discriminator(real_imgs), valid)
                fake_loss = self.criterion(self.discriminator(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                self.opt_d.step()
                # Record losses
                history["g_loss"].append(g_loss.item())
                history["d_loss"].append(d_loss.item())
                if (i + 1) % log_interval == 0:
                    print(f"[Epoch {epoch}/{epochs}] [Batch {i+1}/{len(dataloader)}] "
                          f"D loss: {d_loss.item():.4f}, G loss: {g_loss.item():.4f}")
            # End of epoch summary
            avg_g = sum(history["g_loss"][-len(dataloader):]) / len(dataloader)
            avg_d = sum(history["d_loss"][-len(dataloader):]) / len(dataloader)
            print(f"--- Epoch {epoch} completed | Avg D loss: {avg_d:.4f}, Avg G loss: {avg_g:.4f} ---")
        return history

    def generate_samples(self, num_samples: int = 16):
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=self.device)
            samples = self.generator(z)
        return samples.cpu()
