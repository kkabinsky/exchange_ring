import glob  
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import time
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

# Configure logging for better visibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tadgans_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("tadGANs")

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

if not os.path.exists('results'):
    os.makedirs('results')

##################################################################################
####################### Core functions for anomaly detection #####################
##################################################################################

def test(test_loader, encoder, decoder, critic_x, device):
    """
    Calculate anomaly scores for test data
    
    Args:
        test_loader: DataLoader for test data
        encoder: Encoder model
        decoder: Decoder model
        critic_x: Critic model for input space
        device: Device to run computation on
        
    Returns:
        anomaly_score: Array of anomaly scores
    """
    encoder.eval()
    decoder.eval()
    critic_x.eval()
    
    reconstruction_error = []
    critic_score = []
    signals = []
    
    with torch.no_grad():
        for batch, sample in enumerate(test_loader):
            sample_signal = sample['signal'].to(device)
            signals.extend(sample_signal.cpu().numpy())
            
            encoded = encoder(sample_signal)
            reconstructed_signal = decoder(encoded)
            reconstructed_signal = torch.squeeze(reconstructed_signal)
            
            # Calculate reconstruction error
            for i in range(len(sample_signal)):
                x_original = sample_signal[i].cpu().numpy()
                x_recon = reconstructed_signal[i].cpu().numpy()
                reconstruction_error.append(dtw_reconstruction_error(x_original, x_recon))
            
            # Get critic score
            critic_output = critic_x(sample_signal)
            critic_score.extend(torch.squeeze(critic_output).cpu().numpy())
    
    # Normalize scores
    reconstruction_error = stats.zscore(reconstruction_error)
    critic_score = stats.zscore(critic_score)
    

    # Combine scores
    anomaly_score = reconstruction_error * critic_score
    
    return anomaly_score, signals
###
def test2(test_loader, encoder, decoder, critic_x):
    reconstruction_error = list()
    critic_score = list()
    y_true = list()

    for batch, sample in enumerate(test_loader):
        reconstructed_signal = decoder(encoder(sample['signal']))
        reconstructed_signal = torch.squeeze(reconstructed_signal)

        for i in range(0, 64):
            x_ = reconstructed_signal[i].detach().numpy()
            x = sample['signal'][i].numpy()
            # y_true.append(int(sample['anomaly'][i].detach()))
            reconstruction_error.append(dtw_reconstruction_error(x, x_))
        critic_score.extend(torch.squeeze(critic_x(sample['signal'])).detach().numpy())

    reconstruction_error = stats.zscore(reconstruction_error)
    critic_score = stats.zscore(critic_score)
    anomaly_score = reconstruction_error * critic_score

    return anomaly_score
###
def dtw_reconstruction_error(x, x_):
    """
    Calculate Dynamic Time Warping (DTW) distance between original and reconstructed signals
    
    Args:
        x: Original signal
        x_: Reconstructed signal
        
    Returns:
        DTW distance between x and x_
    """
    n, m = x.shape[0], x_.shape[0]
    dtw_matrix = np.zeros((n+1, m+1))
    for i in range(n+1):
        for j in range(m+1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0

    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(x[i-1] - x_[j-1])
            last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
            dtw_matrix[i, j] = cost + last_min
    return dtw_matrix[n][m]

def detect_anomaly(anomaly_score, threshold_method='statistical', threshold_value=3):
    """
    Detect anomalies based on anomaly scores
    
    Args:
        anomaly_score: Array of anomaly scores
        threshold_method: Method to determine threshold ('statistical', 'manual')
        threshold_value: Threshold parameter (std deviation for statistical, actual value for manual)
        
    Returns:
        is_anomaly: Binary array indicating anomalies
    """
    is_anomaly = np.zeros(len(anomaly_score))
    
    if threshold_method == 'statistical':
        mean = np.mean(anomaly_score)
        std = np.std(anomaly_score)
        threshold = mean + threshold_value * std
        is_anomaly = (anomaly_score > threshold).astype(int)
    elif threshold_method == 'manual':
        is_anomaly = (anomaly_score > threshold_value).astype(int)
    elif threshold_method == 'adaptive':
        # Using adaptive windowing approach
        window_size = len(anomaly_score) // 3
        step_size = len(anomaly_score) // (3 * 10)

        for i in range(0, len(anomaly_score) - window_size, step_size):
            window_elts = anomaly_score[i:i+window_size]
            window_mean = np.mean(window_elts)
            window_std = np.std(window_elts)

            for j, elt in enumerate(window_elts):
                if (window_mean - threshold_value * window_std) < elt < (window_mean + threshold_value * window_std):
                    is_anomaly[i + j] = 0
                else:
                    is_anomaly[i + j] = 1
    
    return is_anomaly

def prune_false_positives(is_anomaly, anomaly_score, change_threshold=0.1):
    """
    Prune false positive anomalies by analyzing sequences of anomalies
    
    Args:
        is_anomaly: Binary array indicating anomalies
        anomaly_score: Array of anomaly scores
        change_threshold: Threshold for pruning
        
    Returns:
        is_anomaly: Updated binary array
    """
    if 1 not in is_anomaly:
        return is_anomaly
        
    seq_details = []
    start_position = None
    max_seq_element = None
    
    # Find sequences of anomalies
    for i in range(len(is_anomaly)):
        if is_anomaly[i] == 1 and (i == 0 or is_anomaly[i-1] == 0):
            # Start of a new sequence
            start_position = i
            max_seq_element = anomaly_score[i]
        elif is_anomaly[i] == 1:
            # Continuing a sequence
            if anomaly_score[i] > max_seq_element:
                max_seq_element = anomaly_score[i]
        
        if is_anomaly[i] == 1 and (i == len(is_anomaly)-1 or is_anomaly[i+1] == 0):
            # End of a sequence
            end_position = i
            seq_details.append([start_position, end_position, max_seq_element, 0])  # 0 indicates not marked for deletion yet

    if not seq_details:
        return is_anomaly
        
    # Sort sequences by max element
    max_elements = [seq[2] for seq in seq_details]
    sorted_indices = np.argsort(max_elements)[::-1]  # Descending order
    
    # Calculate change percentages
    for i in range(1, len(sorted_indices)):
        curr_idx = sorted_indices[i]
        prev_idx = sorted_indices[i-1]
        curr_max = seq_details[curr_idx][2]
        prev_max = seq_details[prev_idx][2]
        change_percent = abs(curr_max - prev_max) / curr_max if curr_max != 0 else 0
        
        if change_percent < change_threshold:
            seq_details[curr_idx][3] = 1  # Mark for deletion
    
    # Delete marked sequences
    for seq in seq_details:
        if seq[3] == 1:
            is_anomaly[seq[0]:seq[1]+1] = 0
    
    return is_anomaly

def evaluate_performance(y_true, y_pred):
    """
    Evaluate performance of anomaly detection
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        metrics: Dictionary of performance metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    
    tp = fp = fn = tn = 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 1 and p == 0:
            fn += 1
        else:
            tn += 1
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    }
    
    return metrics

##################################################################################
########################### Neural Network Models ################################
##################################################################################

class Encoder(nn.Module):
    def __init__(self, signal_shape=100, latent_dim=20):
        super(Encoder, self).__init__()
        self.signal_shape = signal_shape
        self.latent_dim = latent_dim
        
        self.lstm = nn.LSTM(
            input_size=self.signal_shape, 
            hidden_size=self.latent_dim,
            num_layers=1, 
            bidirectional=True,
            batch_first=True
        )
        
        self.dense = nn.Linear(in_features=self.latent_dim*2, out_features=self.latent_dim)

    def forward(self, x):
        x = x.view(-1, 1, self.signal_shape).float()
        x, (hn, cn) = self.lstm(x)
        x = self.dense(x)
        return x

class Decoder(nn.Module):
    def __init__(self, signal_shape=100, latent_dim=20):
        super(Decoder, self).__init__()
        self.signal_shape = signal_shape
        self.latent_dim = latent_dim
        
        self.lstm = nn.LSTM(
            input_size=self.latent_dim, 
            hidden_size=self.latent_dim*2,
            num_layers=2, 
            bidirectional=True,
            batch_first=True
        )
        
        self.dense = nn.Linear(in_features=self.latent_dim*4, out_features=self.signal_shape)

    def forward(self, x):
        x, (hn, cn) = self.lstm(x)
        x = self.dense(x)
        return x

class CriticX(nn.Module):
    def __init__(self, signal_shape=100, hidden_dim=20):
        super(CriticX, self).__init__()
        self.signal_shape = signal_shape
        
        self.model = nn.Sequential(
            nn.Linear(in_features=self.signal_shape, out_features=hidden_dim*2),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=hidden_dim*2, out_features=hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=hidden_dim, out_features=1)
        )

    def forward(self, x):
        x = x.view(-1, self.signal_shape).float()
        return self.model(x)

class CriticZ(nn.Module):
    def __init__(self, latent_dim=20, hidden_dim=10):
        super(CriticZ, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=hidden_dim, out_features=1)
        )

    def forward(self, x):
        return self.model(x.view(-1, x.size(-1)))

##################################################################################
###################### Dataset and Training Functions ############################
##################################################################################

class SignalDataset(Dataset):
    def __init__(self, path, window_size=100, stride=1, normalize=True, anomaly_labels=None):
        """
        Dataset for time series signals
        
        Args:
            path: Path to CSV file
            window_size: Size of the sliding window
            stride: Step size for the sliding window
            normalize: Whether to normalize the signal
            anomaly_labels: Optional column name containing anomaly labels
        """
        df = pd.read_csv(path)
        self.signal_column = 'signal' if 'signal' in df.columns else df.columns[0]
        
        # Use the first column as signal if 'signal' not found
        if self.signal_column != 'signal':
            logger.info(f"Using column '{self.signal_column}' as signal data")
            df = df.rename(columns={self.signal_column: 'signal'})
        
        self.df = df
        self.window_size = window_size
        self.stride = stride
        
        # Extract signal data
        self.signals = []
        self.labels = []
        
        signal_data = df['signal'].values
        
        # Normalize signal if requested
        if normalize:
            signal_data = (signal_data - np.mean(signal_data)) / (np.std(signal_data) + 1e-10)
        
        # Create sliding windows
        for i in range(0, len(signal_data) - window_size + 1, stride):
            self.signals.append(signal_data[i:i+window_size])
            
            # Include anomaly labels if provided
            if anomaly_labels is not None and anomaly_labels in df.columns:
                # Use majority vote within window as the label
                window_labels = df[anomaly_labels].values[i:i+window_size]
                self.labels.append(1 if sum(window_labels) > window_size/2 else 0)
            else:
                self.labels.append(0)  # Default: not anomalous
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        sample = {'signal': torch.tensor(self.signals[idx], dtype=torch.float32)}
        if self.labels:
            sample['anomaly'] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return sample

def wasserstein_loss(y_true, y_pred):
    """Calculate Wasserstein loss"""
    return torch.mean(y_true * y_pred)

def gradient_penalty(critic, real_samples, fake_samples, device):
    """Calculate gradient penalty for WGAN-GP"""
    # Random weight term for interpolation
    alpha = torch.rand(real_samples.size(0), 1, 1).to(device)
    
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    # Calculate critic output for interpolated samples
    d_interpolates = critic(interpolates)
    
    # Get gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Calculate gradient penalty
    gradients = gradients.view(gradients.size(0), -1)
    gradient_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    
    return gradient_penalty

def train_tadgans(
    train_loader, 
    encoder, 
    decoder, 
    critic_x, 
    critic_z,
    optim_enc,
    optim_dec,
    optim_cx,
    optim_cz,
    device,
    n_epochs=20,
    n_critic=5,
    lambda_gp=10,
    checkpoint_interval=10,
    model_dir='models'
):
    """
    Train the tadGANs model
    
    Args:
        train_loader: DataLoader for training data
        encoder, decoder, critic_x, critic_z: Model components
        optim_enc, optim_dec, optim_cx, optim_cz: Optimizers
        device: Device to run computation on
        n_epochs: Number of epochs
        n_critic: Number of critic iterations per generator iteration
        lambda_gp: Gradient penalty coefficient
        checkpoint_interval: Interval for saving model checkpoints
        model_dir: Directory for saving model checkpoints
    """
    logger.info('Starting training')
    
    # For tracking progress
    losses = {
        'critic_x': [],
        'critic_z': [],
        'encoder': [],
        'decoder': []
    }
    
    mse_loss = nn.MSELoss()
    
    # Move models to device
    encoder.to(device)
    decoder.to(device)
    critic_x.to(device)
    critic_z.to(device)
    
    start_time = time.time()
    
    for epoch in range(n_epochs):
        epoch_start = time.time()
        epoch_losses = {k: [] for k in losses.keys()}
        
        # Train critics
        for _ in range(n_critic):
            for batch, sample in enumerate(train_loader):
                # Move data to device
                real_x = sample['signal'].to(device)
                batch_size = real_x.size(0)
                
                # ---------------------
                #  Train Critic X
                # ---------------------
                optim_cx.zero_grad()
                
                # Generate fake samples
                z = torch.randn(batch_size, 1, encoder.latent_dim).to(device)
                fake_x = decoder(z)
                
                # Calculate Wasserstein loss
                real_validity_x = critic_x(real_x)
                fake_validity_x = critic_x(fake_x.detach())
                
                # Gradient penalty
                gp_x = gradient_penalty(critic_x, real_x, fake_x.detach(), device)
                
                # Adversarial loss
                d_loss_x = -torch.mean(real_validity_x) + torch.mean(fake_validity_x) + lambda_gp * gp_x
                
                d_loss_x.backward()
                optim_cx.step()
                epoch_losses['critic_x'].append(d_loss_x.item())
                
                # ---------------------
                #  Train Critic Z
                # ---------------------
                optim_cz.zero_grad()
                
                # Get encoded samples
                real_z = encoder(real_x)
                # Generate random samples
                fake_z = torch.randn(batch_size, 1, encoder.latent_dim).to(device)
                
                # Calculate Wasserstein loss
                real_validity_z = critic_z(real_z)
                fake_validity_z = critic_z(fake_z)
                
                # Gradient penalty
                gp_z = gradient_penalty(critic_z, real_z.detach(), fake_z, device)
                
                # Adversarial loss
                d_loss_z = -torch.mean(real_validity_z) + torch.mean(fake_validity_z) + lambda_gp * gp_z
                
                d_loss_z.backward()
                optim_cz.step()
                epoch_losses['critic_z'].append(d_loss_z.item())
                
        # Train Generator (Encoder-Decoder)
        for batch, sample in enumerate(train_loader):
            real_x = sample['signal'].to(device)
            batch_size = real_x.size(0)
            
            # ---------------------
            #  Train Encoder
            # ---------------------
            optim_enc.zero_grad()
            
            # Encode and reconstruct
            z_enc = encoder(real_x)
            x_rec = decoder(z_enc)
            
            # Reconstruction loss
            rec_loss = mse_loss(real_x, x_rec)
            
            # Adversarial loss (fool critic Z)
            adv_loss_enc = -torch.mean(critic_z(z_enc))
            
            # Total loss
            g_loss_enc = rec_loss + adv_loss_enc
            
            g_loss_enc.backward(retain_graph=True)
            optim_enc.step()
            epoch_losses['encoder'].append(g_loss_enc.item())
            
            # ---------------------
            #  Train Decoder
            # ---------------------
            optim_dec.zero_grad()
            
            # Generate random latent vectors
            z = torch.randn(batch_size, 1, encoder.latent_dim).to(device)
            # Generate signals
            fake_x = decoder(z)
            
            # Re-encode and reconstruct to ensure cycle consistency
            z_enc_rec = encoder(real_x)
            x_rec = decoder(z_enc_rec)
            
            # Reconstruction loss
            rec_loss = mse_loss(real_x, x_rec)
            
            # Adversarial loss (fool critic X)
            adv_loss_dec = -torch.mean(critic_x(fake_x))
            
            # Total loss
            g_loss_dec = rec_loss + adv_loss_dec
            
            g_loss_dec.backward()
            optim_dec.step()
            epoch_losses['decoder'].append(g_loss_dec.item())
        
        # Update average losses
        for k in losses.keys():
            losses[k].append(np.mean(epoch_losses[k]))
        
        # Log progress
        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch+1}/{n_epochs} - Time: {epoch_time:.2f}s - "
                   f"Loss: critic_x={losses['critic_x'][-1]:.4f}, "
                   f"critic_z={losses['critic_z'][-1]:.4f}, "
                   f"encoder={losses['encoder'][-1]:.4f}, "
                   f"decoder={losses['decoder'][-1]:.4f}")
        
        # Save checkpoints
        if (epoch + 1) % checkpoint_interval == 0 or epoch == n_epochs - 1:
            torch.save(encoder.state_dict(), f"{model_dir}/encoder_{epoch+1}.pt")
            torch.save(decoder.state_dict(), f"{model_dir}/decoder_{epoch+1}.pt")
            torch.save(critic_x.state_dict(), f"{model_dir}/critic_x_{epoch+1}.pt")
            torch.save(critic_z.state_dict(), f"{model_dir}/critic_z_{epoch+1}.pt")
            logger.info(f"Models saved at epoch {epoch+1}")
    
    # Save final models
    torch.save(encoder.state_dict(), f"{model_dir}/encoder_final.pt")
    torch.save(decoder.state_dict(), f"{model_dir}/decoder_final.pt")
    torch.save(critic_x.state_dict(), f"{model_dir}/critic_x_final.pt")
    torch.save(critic_z.state_dict(), f"{model_dir}/critic_z_final.pt")
    
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f}s")
    
    return losses

##################################################################################
########################### Utility Functions ###################################
##################################################################################

def plot_results(df, output_path=None, show_plot=True, threshold=None):
    """
    Plot signal and anomaly scores
    
    Args:
        df: DataFrame with signal and anomaly scores
        output_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
        threshold: Anomaly threshold to plot (optional)
    """
    plt.figure(figsize=(14, 10))
    
    # Plot signal
    plt.subplot(3, 1, 1)
    plt.plot(df['signal'], 'b-', linewidth=1)
    plt.title('Original Signal', fontsize=14)
    plt.ylabel('Value', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Plot anomaly score
    plt.subplot(3, 1, 2)
    plt.plot(df['anomaly_score'], 'r-', linewidth=1)
    plt.title('Anomaly Score', fontsize=14)
    plt.ylabel('Score', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    if threshold is not None:
        plt.axhline(y=threshold, color='g', linestyle='--', label=f'Threshold: {threshold:.2f}')
        plt.legend()
    
    # Plot detected anomalies
    plt.subplot(3, 1, 3)
    plt.plot(df['signal'], 'b-', linewidth=1, alpha=0.5)
    
    if 'is_anomaly' in df.columns:
        anomaly_points = df[df['is_anomaly'] == 1]
        plt.scatter(anomaly_points.index, anomaly_points['signal'], 
                   color='red', label='Detected Anomalies', s=30)
    
    plt.title('Signal with Detected Anomalies', fontsize=14)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def excel_to_csv(excel_files, output_dir=None):
    """
    Convert Excel files to CSV format
    
    Args:
        excel_files: List of Excel file paths
        output_dir: Directory to save CSV files (optional)
        
    Returns:
        List of CSV file paths
    """
    csv_files = []
    
    for excel_file in excel_files:
        try:
            # Read Excel file
            df = pd.read_excel(excel_file)
            
            # Find potential signal columns
            signal_col = None
            for col in ['close', 'Close', 'price', 'Price', 'value', 'Value']:
                if col in df.columns:
                    signal_col = col
                    break
            
            if signal_col is None:
                logger.warning(f"No recognizable signal column found in {excel_file}. Using first column.")
                signal_col = df.columns[0]
            
            # Keep only signal column and rename it to 'signal'
            df_signal = df[[signal_col]].rename(columns={signal_col: 'signal'})
            
            # Create CSV filename and path
            base_name = os.path.splitext(os.path.basename(excel_file))[0]
            
            if output_dir:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                csv_path = os.path.join(output_dir, f"{base_name}.csv")
            else:
                csv_path = f"{base_name}.csv"
            
            # Save to CSV
            df_signal.to_csv(csv_path, index=False)
            csv_files.append(csv_path)
            logger.info(f"Converted {excel_file} to {csv_path}")
            
        except Exception as e:
            logger.error(f"Error converting {excel_file}: {str(e)}")
    
    return csv_files

def load_or_train_models(input_file, config, force_train=False):
    """
    Load existing models or train new ones
    
    Args:
        input_file: Path to input data file
        config: Configuration dictionary
        force_train: Whether to force training even if models exist
        
    Returns:
        tuple: (encoder, decoder, critic_x, critic_z)
    """
    # Set up models
    encoder = Encoder(config['signal_shape'], config['latent_dim'])
    decoder = Decoder(config['signal_shape'], config['latent_dim'])
    critic_x = CriticX(config['signal_shape'], config['hidden_dim'])
    critic_z = CriticZ(config['latent_dim'], config['hidden_dim'] // 2)
    
    # Check if models exist
    encoder_path = f"{config['model_dir']}/encoder_final.pt"
    decoder_path = f"{config['model_dir']}/decoder_final.pt"
    critic_x_path = f"{config['model_dir']}/critic_x_final.pt"
    critic_z_path = f"{config['model_dir']}/critic_z_final.pt"
    
    models_exist = (
        os.path.exists(encoder_path) and 
        os.path.exists(decoder_path) and 
        os.path.exists(critic_x_path) and 
        os.path.exists(critic_z_path)
    )
    
    if models_exist and not force_train:
        logger.info("Loading existing models")
        encoder.load_state_dict(torch.load(encoder_path, map_location=config['device']))
        decoder.load_state_dict(torch.load(decoder_path, map_location=config['device']))
        critic_x.load_state_dict(torch.load(critic_x_path, map_location=config['device']))
        critic_z.load_state_dict(torch.load(critic_z_path, map_location=config['device']))
    else:
        logger.info("Training new models")
        # Prepare dataset and data loader
        train_dataset = SignalDataset(
            path=input_file, 
            window_size=config['signal_shape'],
            stride=config['stride'],
            normalize=config['normalize']
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'], 
            shuffle=True,
            drop_last=True
        )
        
        # Set up optimizers
        optim_enc = optim.Adam(encoder.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        optim_dec = optim.Adam(decoder.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        optim_cx = optim.Adam(critic_x.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        optim_cz = optim.Adam(critic_z.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        
        # Train models
        train_tadgans(
            train_loader=train_loader,
            encoder=encoder,
            decoder=decoder,
            critic_x=critic_x,
            critic_z=critic_z,
            optim_enc=optim_enc,
            optim_dec=optim_dec,
            optim_cx=optim_cx,
            optim_cz=optim_cz,
            device=config['device'],
            n_epochs=config['n_epochs'],
            n_critic=config['n_critic'],
            lambda_gp=config['lambda_gp'],
            checkpoint_interval=config['checkpoint_interval'],
            model_dir=config['model_dir']
        )
    
    return encoder, decoder, critic_x, critic_z

def process_file(input_file, output_file, config, force_train=False):
    """
    Process a single file for anomaly detection
    
    Args:
        input_file: Path to input file
        output_file: Path to output file
        config: Configuration dictionary
        force_train: Whether to force training even if models exist
        
    Returns:
        DataFrame with anomaly detection results
    """
    try:
        logger.info(f"Processing {input_file}")
        
        # Load original data
        df = pd.read_csv(input_file)
        
        # Create dataset for testing
        test_dataset = SignalDataset(
            path=input_file,
            window_size=config['signal_shape'],
            stride=config['stride'],
            normalize=config['normalize']
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            drop_last=False
        )
        
        # Load or train models
        encoder, decoder, critic_x, critic_z = load_or_train_models(input_file, config, force_train)
        
        # Move models to device
        encoder.to(config['device'])
        decoder.to(config['device'])
        critic_x.to(config['device'])
        critic_z.to(config['device'])
        
        # Set models to evaluation mode
        encoder.eval()
        decoder.eval()
        critic_x.eval()
        critic_z.eval()
        
        # Calculate anomaly scores
        logger.info("Calculating anomaly scores")
        anomaly_scores, _ = test(test_loader, encoder, decoder, critic_x, config['device'])
        
        # Prepare results DataFrame
        results_df = pd.DataFrame({'signal': df['signal'].values[:len(anomaly_scores)]})
        results_df['anomaly_score'] = anomaly_scores
        
        # Detect anomalies
        threshold = None
        if config['threshold_method'] == 'statistical':
            threshold = np.mean(anomaly_scores) + config['threshold_value'] * np.std(anomaly_scores)
            logger.info(f"Using statistical threshold: {threshold:.4f}")
        elif config['threshold_method'] == 'manual':
            threshold = config['threshold_value']
            logger.info(f"Using manual threshold: {threshold}")
        
        results_df['is_anomaly'] = detect_anomaly(
            anomaly_scores, 
            threshold_method=config['threshold_method'],
            threshold_value=config['threshold_value']
        )
        
        # Prune false positives if enabled
        if config['prune_false_positives']:
            logger.info("Pruning false positives")
            results_df['is_anomaly'] = prune_false_positives(
                results_df['is_anomaly'].values,
                results_df['anomaly_score'].values,
                config['prune_threshold']
            )
        
        # Generate plots
        if config['generate_plots']:
            plot_path = f"{os.path.splitext(output_file)[0]}_plot.png"
            plot_results(
                results_df,
                output_path=plot_path,
                show_plot=config['show_plots'],
                threshold=threshold
            )
        
        # Save results to CSV
        results_df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
        
        # Export to Excel if requested
        if config['export_excel']:
            excel_path = f"{os.path.splitext(output_file)[0]}.xlsx"
            results_df.to_excel(excel_path, index=False)
            logger.info(f"Results exported to Excel: {excel_path}")
        
        return results_df
        
    except Exception as e:
        logger.error(f"Error processing {input_file}: {str(e)}")
        raise e

def batch_process_files(input_files, output_dir, config):
    """
    Process multiple files for anomaly detection
    
    Args:
        input_files: List of input file paths
        output_dir: Directory for output files
        config: Configuration dictionary
        
    Returns:
        Dictionary mapping input files to summary metrics
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    results = {}
    
    for input_file in input_files:
        try:
            # Create output filename
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            output_file = os.path.join(output_dir, f"{base_name}_results.csv")
            
            # Process file
            logger.info(f"Processing {input_file}")
            df = process_file(input_file, output_file, config)
            
            # Calculate summary metrics
            anomaly_count = df['is_anomaly'].sum()
            total_points = len(df)
            anomaly_percentage = (anomaly_count / total_points) * 100
            
            metrics = {
                'total_points': total_points,
                'anomaly_count': anomaly_count,
                'anomaly_percentage': anomaly_percentage,
                'mean_score': df['anomaly_score'].mean(),
                'max_score': df['anomaly_score'].max(),
                'output_file': output_file
            }
            
            results[input_file] = metrics
            
        except Exception as e:
            logger.error(f"Error processing {input_file}: {str(e)}")
            results[input_file] = {'error': str(e)}
    
    # Create summary report
    summary_df = pd.DataFrame.from_dict(results, orient='index')
    summary_path = os.path.join(output_dir, "batch_summary.csv")
    summary_df.to_csv(summary_path)
    
    # Export to Excel if requested
    if config['export_excel']:
        excel_path = os.path.join(output_dir, "batch_summary.xlsx")
        summary_df.to_excel(excel_path)
    
    return results

def performance_test(test_file, labeled_anomalies_file, config):
    """
    Run a performance test on the anomaly detection model
    
    Args:
        test_file: Path to test data file
        labeled_anomalies_file: Path to file with labeled anomalies
        config: Configuration dictionary
        
    Returns:
        Dictionary of performance metrics
    """
    # Load test data
    df = pd.read_csv(test_file)
    
    # Load labeled anomalies (should contain a column 'is_anomaly' with 1/0 values)
    labeled_df = pd.read_csv(labeled_anomalies_file)
    
    # Process file for anomaly detection
    results_df = process_file(
        test_file,
        f"{os.path.splitext(test_file)[0]}_perf_test.csv",
        config
    )
    
    # Align results with labeled anomalies
    min_len = min(len(results_df), len(labeled_df))
    y_true = labeled_df['is_anomaly'].values[:min_len]
    y_pred = results_df['is_anomaly'].values[:min_len]
    
    # Calculate performance metrics
    metrics = evaluate_performance(y_true, y_pred)
    
    # Generate performance plot
    plt.figure(figsize=(14, 10))
    
    # Plot signal
    plt.subplot(2, 1, 1)
    plt.plot(df['signal'].values[:min_len], 'b-', linewidth=1, alpha=0.5)
    
    # Plot true anomalies
    true_anomalies = np.where(y_true == 1)[0]
    plt.scatter(true_anomalies, df['signal'].values[true_anomalies], color='g', label='True Anomalies', s=30, marker='*')
    
    # Plot detected anomalies
    detected_anomalies = np.where(is_anomaly == 1)[0]
    plt.scatter(detected_anomalies, df['signal'].values[detected_anomalies], color='r', label='Detected Anomalies', s=25, marker='o')
    
    plt.title('Anomaly Detection Performance', fontsize=14)
    plt.ylabel('Signal Value', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot confusion statistics
    plt.subplot(2, 1, 2)
    
    # Create confusion matrix visualization
    labels = ['Normal', 'Anomaly']
    cm = np.array([
        [metrics['true_negatives'], metrics['false_positives']],
        [metrics['false_negatives'], metrics['true_positives']]
    ])
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontsize=14)
    plt.colorbar()
    
    # Add text annotations
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{os.path.splitext(test_file)[0]}_performance.png", dpi=300, bbox_inches='tight')
    
    # Log metrics
    logger.info(f"Performance metrics: {metrics}")
    
    # Save metrics to file
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f"{os.path.splitext(test_file)[0]}_metrics.csv", index=False)
    
    if config['export_excel']:
        metrics_df.to_excel(f"{os.path.splitext(test_file)[0]}_metrics.xlsx", index=False)
    
    return metrics

##################################################################################
################################## Main Functions ################################
##################################################################################

def main():
    """Main function for running tadGANs anomaly detection"""
    # Default configuration
    config = {
        # Model parameters
        'signal_shape': 100,        # Window size for time series
        'latent_dim': 100,           # Latent space dimension
        'hidden_dim': 40,           # Hidden layer dimension
        
        # Training parameters
        'batch_size': 64,           # Batch size
        'lr': 1e-5,                 # Learning rate
        'n_epochs': 1000,            # Number of epochs
        'n_critic': 5,              # Number of critic iterations per generator iteration
        'lambda_gp': 10,            # Gradient penalty coefficient
        'checkpoint_interval': 10,  # Interval for saving checkpoints
        
        # Data parameters
        'stride': 1,                # Stride for sliding window
        'normalize': True,          # Whether to normalize signals
        
        # Anomaly detection parameters
        'threshold_method': 'statistical',  # 'statistical', 'manual', or 'adaptive'
        'threshold_value': 2.0,             # Threshold parameter
        'prune_false_positives': True,      # Whether to prune false positives
        'prune_threshold': 0.2, #0.1             # Threshold for pruning
        
        # Output parameters
        'generate_plots': True,     # Whether to generate plots
        'show_plots': False,        # Whether to show plots
        'export_excel': True,       # Whether to export results to Excel
        
        # Directory parameters
        'model_dir': 'models',      # Directory for model checkpoints
        'output_dir': 'results',    # Directory for output files
        
        # Device parameters
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }
    
    logger.info(f"Using device: {config['device']}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(config['output_dir']):
        os.makedirs(config['output_dir'])
    
    # Get list of Excel files
    excel_files = glob.glob('*.xlsx')
    
    if not excel_files:
        logger.warning("No Excel files found in current directory!")
        
        # Check if there are CSV files instead
        csv_files = glob.glob('*.csv')
        if csv_files:
            logger.info(f"Found {len(csv_files)} CSV files: {csv_files}")
        else:
            logger.error("No input files found!")
            return
    else:
        logger.info(f"Found {len(excel_files)} Excel files: {excel_files}")
        
        # Convert Excel files to CSV
        csv_files = excel_to_csv(excel_files, output_dir='csv_data')
    
    # Process each CSV file
    for csv_file in csv_files:
        stock_name = os.path.splitext(os.path.basename(csv_file))[0]
        output_file = os.path.join(config['output_dir'], f'{stock_name}_results.csv')
        
        try:
            process_file(csv_file, output_file, config)
            logger.info(f"Successfully processed {csv_file}")
        except Exception as e:
            logger.error(f"Error processing {csv_file}: {str(e)}")
            continue

#def run_with_performance_test(test_file, labeled_file):
 #   """Run with performance testing on labeled data"""
    # Default configuration
  #  config = {
   #     'signal_shape': 100,
    #    'latent_dim': 20,
     #   'hidden_dim': 40,
      #  'batch_size': 64,
       # 'lr': 1e-5,
        #'n_epochs': 1000,
        #'n_critic': 5,
        #'lambda_gp': 10,
        #'checkpoint_interval': 10,
        #'stride': 1,
        #'normalize': True,
        #'threshold_method': 'statistical',
        #'threshold_value': 3.0,
        #'prune_false_positives': True,
        #'prune_threshold': 0.1,
        #'generate_plots': True,
        #'show_plots': False,
        #'export_excel': True,
       # 'model_dir': 'models',
      #  'output_dir': 'results',
     #   'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #}
    
    # Run performance test
#    performance_test(test_file, labeled_file, config)

if __name__ == "__main__":
    main()
