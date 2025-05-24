import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_steps, dropout_rate=0.2):
        super(Encoder, self).__init__()
        self.num_steps = num_steps
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.skip_layers = nn.ModuleList()
        
        print(f"Encoder: input_dim={input_dim}, hidden_dim={hidden_dim}")
        # Initial layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.bns.append(nn.LayerNorm(hidden_dim))  # Replace BatchNorm with LayerNorm
        
        # Hidden layers and skip connections
        if num_steps > 1:
            for i in range(1, num_steps):
                in_dim = hidden_dim // (2 ** (i - 1))
                out_dim = hidden_dim // (2 ** i)
                print(f"Encoder, Layer number {i}: in_dim={in_dim}, out_dim={out_dim}")
                self.layers.append(nn.Linear(in_dim, out_dim))
                self.bns.append(nn.LayerNorm(out_dim))  # Replace BatchNorm with LayerNorm
                self.skip_layers.append(nn.Linear(in_dim, out_dim))
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_mu = nn.Linear(hidden_dim // (2 ** (num_steps - 1)), hidden_dim // (2 ** (num_steps - 1)))
        self.fc_logvar = nn.Linear(hidden_dim // (2 ** (num_steps - 1)), hidden_dim // (2 ** (num_steps - 1)))
    
    def forward(self, x):
        for i in range(self.num_steps):
            x = F.relu(self.bns[i](self.layers[i](x)))
            if i > 0:
                x_skip = self.skip_layers[i - 1](x_prev)
                x = x + x_skip  # Add skip connection
            x_prev = x  # Store previous layer output for skip connection
            x = self.dropout(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_steps, dropout_rate=0.2):
        super(Decoder, self).__init__()
        self.num_steps = num_steps
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.skip_layers = nn.ModuleList()
        
        assert hidden_dim >= 2 ** (num_steps - 1), "hidden_dim too small for num_steps"

        # Initial layer
        if num_steps > 1:
            self.layers.append(nn.Linear(hidden_dim // (2 ** (num_steps - 1)), hidden_dim // (2 ** (num_steps - 2))))
            self.bns.append(nn.LayerNorm(hidden_dim // (2 ** (num_steps - 2))))
            print(f"Decoder Initial layer: in_dim={hidden_dim // (2 ** (num_steps - 1))}, out_dim={hidden_dim // (2 ** (num_steps - 2))}")
        
        # Hidden layers and skip connections
        if num_steps > 1:
            for i in range(1, num_steps - 1):
                in_dim = hidden_dim // (2 ** (num_steps - i - 1))
                out_dim = hidden_dim // (2 ** (num_steps - i - 2))
                print(f"Decoder, Layer number {i}: in_dim={in_dim}, out_dim={out_dim}")
                self.layers.append(nn.Linear(in_dim, out_dim))
                self.bns.append(nn.LayerNorm(out_dim))  # Replace BatchNorm with LayerNorm
                self.skip_layers.append(nn.Linear(in_dim, out_dim))
        
        # Final layer
        self.layers.append(nn.Linear(hidden_dim, input_dim))
        self.dropout = nn.Dropout(dropout_rate)
        print(f"Decoder Final layer: in_dim={hidden_dim}, out_dim={input_dim}")

    def forward(self, x):
        for i in range(self.num_steps - 1):
            x = F.relu(self.bns[i](self.layers[i](x)))
            if i > 0:
                x_skip = self.skip_layers[i - 1](x_prev)
                x = x + x_skip  # Add skip connection
            x_prev = x  # Store previous layer output for skip connection
            x = self.dropout(x)
        x = self.layers[-1](x)  # Final layer without ReLU and dropout
        return x

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_steps, device='cpu', dropout_rate=0.2):
        super(AutoEncoder, self).__init__()

        print(f"AutoEncoder: input_dim={input_dim}, hidden_dim={hidden_dim}, num_steps={num_steps}, dropout_rate={dropout_rate}")
        
        self.encoder = Encoder(input_dim, hidden_dim, num_steps, dropout_rate).to(device)
        self.decoder = Decoder(input_dim, hidden_dim, num_steps, dropout_rate).to(device)
        self.device = device

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = x.to(self.device)
        batch_size, time_steps, neof = x.shape
        decoded = torch.zeros_like(x, device=self.device)
        for t in range(time_steps):
            mu, logvar = self.encoder(x[:, t, :])
            z = self.reparameterize(mu, logvar)
            decoded[:, t, :] = self.decoder(z)
        return decoded, mu, logvar

def train_autoencoder(
    model,
    train_loader,
    val_loader,
    num_epochs=20,
    device="cuda",
    lr=1e-3,
    criterion=None,
    clip_value=1.0,  # New parameter for gradient clipping
    patience=8,  # New parameter for early stopping
    scheduler_step_size=5,  # New parameter for scheduler step size
    scheduler_gamma=0.5,  # New parameter for scheduler gamma
    sign_mismatch_weight=0.5  # New parameter for sign mismatch penalty weight
):
    if criterion is None:
        criterion = nn.MSELoss()
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, amsgrad=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for src, tgt, pasft, futft in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            pasft, futft = pasft.to(device), futft.to(device)
            X = src
            decoded, mu, logvar = model(X)
            recon_loss = criterion(decoded, X)
            
            # Calculate KL divergence
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss /= X.size(0) * X.size(1) * X.size(2)  # Normalize by batch size, time steps, and features
            
            # Calculate sign mismatch penalty
            # sign_mismatch_penalty = torch.mean((torch.sign(X) != torch.sign(decoded)).float())
            
            # Combine losses
            loss = recon_loss + kl_loss #+ sign_mismatch_weight * sign_mismatch_penalty

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_value)  # Gradient clipping
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for src, _, _, _ in val_loader:
                X = src.to(device)
                decoded, mu, logvar = model(X)
                recon_loss = criterion(decoded, X)
                
                # Calculate KL divergence
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                kl_loss /= X.size(0) * X.size(1) * X.size(2)  # Normalize by batch size, time steps, and features
                
                # Calculate sign mismatch penalty
                # sign_mismatch_penalty = torch.mean((torch.sign(X) != torch.sign(decoded)).float())
                
                # Combine losses
                loss = recon_loss + kl_loss #+ sign_mismatch_weight * sign_mismatch_penalty
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}")
        
        # Early stopping
        # Check loss only on training
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
        
        # Step the scheduler
        scheduler.step()
        print(f"Learning rate: {scheduler.get_last_lr()[0]}")