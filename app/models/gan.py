import torch
import torch.nn as nn

class Generator(nn.Module):
    """Improved Conditional Generator for MNIST (28x28).
    Takes a noise vector and class label, outputs a 1x28x28 image.
    Uses batch normalization and deeper architecture for better quality.
    """
    def __init__(self, latent_dim: int = 100, num_classes: int = 10):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Embedding for class label (increased dimension for better conditioning)
        self.label_emb = nn.Embedding(num_classes, 100)
        
        # Generator network: Input is latent_dim + 100 (embedded label)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + 100, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            
            nn.Linear(512, 28 * 28),
            nn.Tanh()  # output in [-1, 1]
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        if labels is None:
            # Unconditional generation - random labels
            labels = torch.randint(0, self.num_classes, (z.size(0),), device=z.device)
        
        # Embed labels
        label_emb = self.label_emb(labels)
        
        # Concatenate noise and embedded label
        z_cond = torch.cat([z, label_emb], dim=1)
        
        # Generate image
        img = self.model(z_cond)
        img = img.view(z.size(0), 1, 28, 28)
        return img

class Discriminator(nn.Module):
    """Improved Conditional Discriminator for MNIST.
    Takes an image and class label, outputs a probability (real/fake).
    Uses batch normalization and dropout for regularization.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        
        # Embedding for class label (increased dimension)
        self.label_emb = nn.Embedding(num_classes, 100)
        
        # Discriminator network: Input is 28*28 + 100 (embedded label)
        self.model = nn.Sequential(
            nn.Linear(28 * 28 + 100, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, img: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        # Flatten image
        img_flat = img.view(img.size(0), -1)
        
        if labels is None:
            # If no labels provided, assume random
            labels = torch.randint(0, self.num_classes, (img.size(0),), device=img.device)
        
        # Embed labels
        label_emb = self.label_emb(labels)
        
        # Concatenate image and embedded label
        img_cond = torch.cat([img_flat, label_emb], dim=1)
        
        return self.model(img_cond)

class GANTrainer:
    """Improved training loop for conditional GAN with label smoothing and better techniques.

    Args:
        generator: Generator model.
        discriminator: Discriminator model.
        device: torch device (cpu or cuda).
        latent_dim: Dimensionality of the noise vector.
        lr: Learning rate for both optimizers.
    """
    def __init__(self, generator: Generator, discriminator: Discriminator, device: torch.device, latent_dim: int = 100, lr: float = 0.0002):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device
        self.latent_dim = latent_dim
        # Use label smoothing: real labels ~0.9, fake labels ~0.1
        self.real_label_smooth = 0.9
        self.fake_label = 0.0
        
        self.criterion = nn.BCELoss()
        self.opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    def train(self, dataloader, epochs: int = 50, log_interval: int = 100):
        """Train the conditional GAN.
        
        Args:
            dataloader: DataLoader with (images, labels) tuples
            epochs: Number of training epochs
            log_interval: Log every N batches
        """
        self.generator.train()
        self.discriminator.train()
        history = {"g_loss": [], "d_loss": [], "epoch_g_loss": [], "epoch_d_loss": []}
        
        for epoch in range(1, epochs + 1):
            epoch_g_losses = []
            epoch_d_losses = []
            
            for i, (real_imgs, labels) in enumerate(dataloader):
                batch_size = real_imgs.size(0)
                real_imgs = real_imgs.to(self.device)
                labels = labels.to(self.device)
                
                # Label smoothing
                valid = torch.ones(batch_size, 1, device=self.device) * self.real_label_smooth
                fake = torch.ones(batch_size, 1, device=self.device) * self.fake_label
                
                # -----------------
                #  Train Generator
                # -----------------
                self.opt_g.zero_grad()
                z = torch.randn(batch_size, self.latent_dim, device=self.device)
                gen_imgs = self.generator(z, labels)
                # Generator wants discriminator to think it's real
                g_loss = self.criterion(self.discriminator(gen_imgs, labels), valid)
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
                self.opt_g.step()
                
                # ---------------------
                #  Train Discriminator
                # ---------------------
                self.opt_d.zero_grad()
                # Real images should be classified as real
                real_loss = self.criterion(self.discriminator(real_imgs, labels), valid)
                # Fake images should be classified as fake
                z = torch.randn(batch_size, self.latent_dim, device=self.device)
                gen_imgs = self.generator(z, labels)
                fake_loss = self.criterion(self.discriminator(gen_imgs.detach(), labels), fake)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
                self.opt_d.step()
                
                # Record losses
                g_loss_item = g_loss.item()
                d_loss_item = d_loss.item()
                history["g_loss"].append(g_loss_item)
                history["d_loss"].append(d_loss_item)
                epoch_g_losses.append(g_loss_item)
                epoch_d_losses.append(d_loss_item)
                
                if (i + 1) % log_interval == 0:
                    print(f"[Epoch {epoch}/{epochs}] [Batch {i+1}/{len(dataloader)}] "
                          f"D loss: {d_loss_item:.4f}, G loss: {g_loss_item:.4f}")
            
            # End of epoch summary
            avg_g = sum(epoch_g_losses) / len(epoch_g_losses)
            avg_d = sum(epoch_d_losses) / len(epoch_d_losses)
            history["epoch_g_loss"].append(avg_g)
            history["epoch_d_loss"].append(avg_d)
            print(f"--- Epoch {epoch}/{epochs} | D loss: {avg_d:.6f}, G loss: {avg_g:.6f} ---")
        
        return history

    def generate_samples(self, num_samples: int = 16, digit_class: int = None):
        """Generate samples from the generator.
        
        Args:
            num_samples: Number of samples to generate.
            digit_class: Specific digit class to generate (0-9). If None, random classes.
        """
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=self.device)
            
            if digit_class is not None:
                # Generate specific digit class
                labels = torch.full((num_samples,), digit_class, dtype=torch.long, device=self.device)
            else:
                # Random labels
                labels = torch.randint(0, 10, (num_samples,), device=self.device)
            
            samples = self.generator(z, labels)
        return samples.cpu()

    def save_checkpoint(self, filepath: str):
        """Save generator and discriminator models to a checkpoint file.
        
        Args:
            filepath: Path to save the checkpoint.
        """
        checkpoint = {
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'opt_g_state_dict': self.opt_g.state_dict(),
            'opt_d_state_dict': self.opt_d.state_dict(),
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def save_models(self, generator_path: str, discriminator_path: str):
        """Save generator and discriminator models separately.
        
        Args:
            generator_path: Path to save the generator model.
            discriminator_path: Path to save the discriminator model.
        """
        torch.save(self.generator.state_dict(), generator_path)
        torch.save(self.discriminator.state_dict(), discriminator_path)
        print(f"Generator saved to {generator_path}")
        print(f"Discriminator saved to {discriminator_path}")
