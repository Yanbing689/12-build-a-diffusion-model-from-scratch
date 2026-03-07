# Step 1: Import libraries and configure 

# pip install transformers sentence-transformers faiss-cpu langchain

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
IMG_SIZE = 32 # We'll resize MNIST images to 32x32
TIMESTEPS = 300 # Number of steps in the diffusion process
LEARNING_RATE = 1e-3
EPOCHS = 20
OUTPUT_DIR = "diffusion_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_data_loader():
    """
    Prepares and returns the MNIST DataLoader.
    """
    # Define transformations for the images
    # 1. Resize to IMG_SIZE
    # 2. Convert to Tensor
    # 3. Normalize to [-1, 1] range
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])

    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader


# Step 2: The Forward Process (Adding the Noise)
def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    """
    Creates a linear variance schedule.

    Args:
        timesteps (int): The number of timesteps.

    Returns:
        torch.Tensor: A tensor of beta values.
    """
    return torch.linspace(start, end, timesteps)


# Get the beta schedule
betas = linear_beta_schedule(timesteps=TIMESTEPS).to(DEVICE)

# Calculate alphas
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0) # Cumulative product of alphas
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0) # Previous cumulative product

# Calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# Calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


def extract(a, t, x_shape):
    """
    Extracts the values from 'a' at the indices 't' and reshapes it to
    match the batch dimension of 'x'.
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(DEVICE)


def q_sample(x_start, t, noise=None):
    """
    Forward diffusion process: adds noise to an image.

    Args:
        x_start (torch.Tensor): The initial image (x_0).
        t (torch.Tensor): The timestep index.
        noise (torch.Tensor, optional): The noise to add. If None, generated randomly.

    Returns:
        torch.Tensor: The noised image at timestep t.
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)

    # Equation for noising: x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


# Step 3: Building the Brain of the Model (The U-Net)
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(...,) + (None,) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Up sample
        return self.transform(h)


class SimpleUnet(nn.Module):
    def __init__(self):
        super().__init__()
        image_channels = 1 # MNIST is grayscale
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 1
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsampling
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], \
                                    time_emb_dim) \
                    for i in range(len(down_channels)-1)])
        # Upsampling
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                        time_emb_dim, up=True) \
                    for i in range(len(up_channels)-1)])

        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        return self.output(x)
   
    
# Step 4: The Reverse Process (Denoising)
@torch.no_grad()
def p_sample(model, x, t, t_index):
    """
    Performs one step of the reverse diffusion process (sampling).

    Args:
        model (nn.Module): The U-Net model.
        x (torch.Tensor): The current noisy image (x_t).
        t (torch.Tensor): The current timestep.
        t_index (int): The index of the current timestep.

    Returns:
        torch.Tensor: The de-noised image for the previous timestep (x_{t-1}).
    """
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def p_sample_loop(model, shape):
    """
    The full sampling loop to generate an image from noise.

    Args:
        model (nn.Module): The U-Net model.
        shape (tuple): The shape of the image to generate (e.g., [batch_size, channels, H, W]).

    Returns:
        torch.Tensor: The final generated image.
    """
    img = torch.randn(shape, device=DEVICE)
    imgs = []

    for i in tqdm(reversed(range(0, TIMESTEPS)), desc="Sampling loop", total=TIMESTEPS):
        t = torch.full((shape[0],), i, device=DEVICE, dtype=torch.long)
        img = p_sample(model, img, t, i)
        # Optional: save intermediate steps
        # if i % 50 == 0:
        #     imgs.append(img.cpu())
    return img


# Step 5: Training the Model
def get_loss(model, x_start, t):
    """
    Calculates the loss for a given batch.
    """
    # 1. Generate random noise
    noise = torch.randn_like(x_start)

    # 2. Get the noised image at timestep t
    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)

    # 3. Get the model's noise prediction
    predicted_noise = model(x_noisy, t)

    # 4. Calculate the loss between the actual noise and predicted noise
    loss = F.l1_loss(noise, predicted_noise) # L1 loss is common and works well

    return loss

def train():
    dataloader = get_data_loader()
    model = SimpleUnet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        # Use tqdm for a nice progress bar
        loop = tqdm(dataloader, leave=True)
        for batch_idx, (images, _) in enumerate(loop):
            images = images.to(DEVICE)

            # Sample a random timestep for each image in the batch
            t = torch.randint(0, TIMESTEPS, (images.shape[0],), device=DEVICE).long()

            # Calculate loss
            loss = get_loss(model, images, t)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update progress bar
            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(loss=loss.item())

        # --- After each epoch, generate and save a sample image ---
        print(f"Epoch {epoch+1} completed. Generating sample image...")
        num_images_to_sample = 16
        sample_shape = (num_images_to_sample, 1, IMG_SIZE, IMG_SIZE)

        # Generate the images
        generated_images = p_sample_loop(model, sample_shape)

        # Denormalize from [-1, 1] to [0, 1] for saving
        generated_images = (generated_images + 1) * 0.5

        # Save the image grid
        save_image(generated_images, os.path.join(OUTPUT_DIR, f"epoch_{epoch+1}_sample.png"), nrow=4)
        print(f"Sample image saved for epoch {epoch+1}.")

    # Save the final model
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "diffusion_mnist.pth"))
    print("Training finished and model saved.")
    
if __name__ == "__main__":
    print("Starting Diffusion Model Training...")
    print(f"Device: {DEVICE}")
    print(f"Timesteps: {TIMESTEPS}")
    print(f"Batch Size: {BATCH_SIZE}")

    # Run the training process
    train()

    print("All steps complete. You can find generated images and the saved model in the 'diffusion_outputs' directory.")    
