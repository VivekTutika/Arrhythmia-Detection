"""import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from train_dsnn import train_dsnn

class DSNN(nn.Module):
    def __init__(self, input_channels=1, sequence_length=24, num_classes=6, max_channels=32):
        super(DSNN, self).__init__()

        # Ensure input_channels doesn't exceed max_channels
        self.input_channels = min(input_channels, max_channels)
        self.sequence_length = sequence_length
        
        # CNN Layer 1: 3 1x7 filters with 1x2 max pooling
        self.conv1 = nn.Conv2d(self.input_channels, 3, kernel_size=(1, 7), padding=(0, 3))
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2))
        
        # CNN Layer 2: 1x1 convolutional filters
        self.conv2 = nn.Conv2d(3, 3, kernel_size=(1, 1))
        
        # CNN Layer 3: 1x7 filters with 1x3 max pooling
        self.conv3 = nn.Conv2d(3, 3, kernel_size=(1, 7), padding=(0, 3))
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 3))
        
        # Calculate the size of flattened features
        self.flat_size = None
        self._calculate_flat_size(self.input_channels, sequence_length)
        print(f"Initial flat_size in __init__: {self.flat_size}")
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(self.flat_size, 14)
        self.fc2 = nn.Linear(14, num_classes)
        
    def _calculate_flat_size(self, channels, seq_len):
        # Helper method to calculate flattened size
        x = torch.randn(1, channels, 1, seq_len)
        x = self.pool1(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool3(F.relu(self.conv3(x)))
        self.flat_size = x.numel()
        print(f"Calculated flat_size: {self.flat_size}")
    
    def forward(self, x):
        # Check and reshape input if needed
        if x.dim() == 4 and x.size(1) != self.input_channels:
            # If x is [batch_size, 32, 2, 24], transpose to [batch_size, 2, 1, 24]
            x = x.permute(0, 2, 1, 3)
            # If needed, reshape further
            x = x.reshape(x.size(0), self.input_channels, 1, -1)
        # CNN layers with ReLU activation and max pooling
        x = self.pool1(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool3(F.relu(self.conv3(x)))
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return F.softmax(x, dim=1)

class DSNNSystem:
    def __init__(self, model):
        self.model = model
        
    def shift_sequence(self, sequence, shift_right=True):
        #removed 3 quotations
        Shifts the input sequence one step right or left
        sequence: tensor of shape (batch_size, channels, 1, sequence_length)
        #removed 3 quotations
        if shift_right:
            return torch.roll(sequence, shifts=1, dims=-1)
        return torch.roll(sequence, shifts=-1, dims=-1)
    
    def voter(self, original_pred, shifted_pred):
        #removed 3 quotations
        Implements voting mechanism to combine predictions
        from original and shifted sequences
        #removed 3 quotations
        combined_pred = (original_pred + shifted_pred) / 2
        return torch.argmax(combined_pred, dim=1)
    
    def process_ecg(self, ecg_sequence):
        #removed 3 quotations
        Process a single ECG sequence through the DSNN system
        #removed 3 quotations
        # Ensure model is in evaluation mode
        self.model.eval()
        
        with torch.no_grad():
            # Original sequence prediction
            original_pred = self.model(ecg_sequence)
            
            # Shifted sequence prediction
            shifted_sequence = self.shift_sequence(ecg_sequence)
            shifted_pred = self.model(shifted_sequence)
            
            # Get final prediction through voting
            final_prediction = self.voter(original_pred, shifted_pred)
            
        return final_prediction

#removed 3 quotationsdef train_dsnn(model, train_loader, num_epochs=2000, learning_rate=0.001, device='cuda'):
    
    #Training function for DSNN

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    model.to(device)
    model.train()
    model._calculate_flat_size(32, 24)
    model.fc1 = nn.Linear(model.flat_size, 14)
    
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Generate shifted data
            shifted_data = torch.roll(data, shifts=1, dims=-1)
            
            # Forward pass with original and shifted data
            optimizer.zero_grad()
            output_original = model(data)
            output_shifted = model(shifted_data)
            
            # Combine losses from both original and shifted data
            loss = criterion(output_original, target) + criterion(output_shifted, target)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

# Example usage
def prepare_data(ecg_data, sequence_length=24):
    #Prepare ECG data for DSNN (example function)
    # Resample to 360 Hz if necessary
    # Extract sequences of length 24 around R-peaks
    # Return prepared data in the format (batch_size, 2, 1, 24)
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# First copy and paste the DSNN and DSNNSystem classes from the previous code here
# [Previous DSNN and DSNNSystem class definitions should go here]#removed 3 quotations

# Create sample ECG data
def create_sample_ecg_data(num_samples=100):
    #Generate synthetic ECG-like data for testing
    # Create time points
    t = np.linspace(0, 23, 24)
    
    # Generate sample data for both leads
    samples = []
    labels = []
    
    for i in range(num_samples):
        # Create synthetic ECG-like signals
        lead1 = np.sin(t + np.random.rand()) + np.random.normal(0, 0.1, 24)
        lead2 = np.cos(t + np.random.rand()) + np.random.normal(0, 0.1, 24)
        
        # Combine leads
        sample = np.stack([lead1, lead2])
        samples.append(sample)
        
        # Generate random labels (0-5 for 6 classes)
        labels.append(np.random.randint(0, 6))
    
    return np.array(samples), np.array(labels)

#removed 3 quotations# Create DataLoader
def create_data_loader(samples, labels, batch_size=32):
    #Convert numpy arrays to PyTorch DataLoader
    # Convert to torch tensors
    X = torch.FloatTensor(samples).unsqueeze(2)  # Add channel dimension
    y = torch.LongTensor(labels)
    
    # Create TensorDataset
    dataset = torch.utils.data.TensorDataset(X, y)
    
    # Create DataLoader
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return loader#removed 3 quotations

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Set number of channels for this run
    num_channels = 12  # Change this to test different numbers of channels
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create sample data
    print("Generating sample data with {num_channels} channels...")
    samples, labels = create_sample_ecg_data(1000, num_channels = num_channels)
    train_loader = create_data_loader(samples, labels)
    
    # Initialize model
    print("Initializing DSNN model with {num_channels} input channels...")
    model = DSNN(input_channels=num_channels, sequence_length=24, num_classes=6)
    dsnn_system = DSNNSystem(model)
    
    # Train model
    print("Training model...")
    train_dsnn(model, train_loader, val_loader=None, num_epochs=2000, learning_rate=0.001, device='cuda', class_names=None)
    # Reduced epochs for demonstration
    
    # Test the model with a single sample
    print("\nTesting model with a single sample...")
    test_sample = torch.FloatTensor(samples[0:1]).unsqueeze(2).to(device)
    prediction = dsnn_system.process_ecg(test_sample)
    print(f"Prediction for test sample: Class {prediction.item()}")
    
    # Visualize sample ECG data
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(samples[0][0], label='Lead 1')
    plt.title('Lead 1 ECG Signal')
    plt.xlabel('Time point')
    plt.ylabel('Amplitude')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(samples[0][1], label='Lead 2')
    plt.title('Lead 2 ECG Signal')
    plt.xlabel('Time point')
    plt.ylabel('Amplitude')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class DSNN(nn.Module):
    def __init__(self, input_channels=1, sequence_length=24, num_classes=6, max_channels=32):
        super(DSNN, self).__init__()

        # Ensure input_channels is within valid range (1 to 32)
        self.input_channels = max(1, min(input_channels, max_channels))
        self.sequence_length = sequence_length
        
        # CNN Layer 1: 3 1x7 filters with 1x2 max pooling
        self.conv1 = nn.Conv2d(self.input_channels, 3, kernel_size=(1, 7), padding=(0, 3))
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2))
        
        # CNN Layer 2: 1x1 convolutional filters
        self.conv2 = nn.Conv2d(3, 3, kernel_size=(1, 1))
        
        # CNN Layer 3: 1x7 filters with 1x3 max pooling
        self.conv3 = nn.Conv2d(3, 3, kernel_size=(1, 7), padding=(0, 3))
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 3))
        
        # Calculate the size of flattened features
        self.flat_size = None
        self._calculate_flat_size()
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(self.flat_size, 14)
        self.fc2 = nn.Linear(14, num_classes)
        
    def _calculate_flat_size(self):
        # Helper method to calculate flattened size based on current input_channels
        x = torch.randn(1, self.input_channels, 1, self.sequence_length)
        x = self.pool1(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool3(F.relu(self.conv3(x)))
        self.flat_size = x.numel()
        print(f"Calculated flat_size for {self.input_channels} channels: {self.flat_size}")
    
    def forward(self, x):
        # Handle input shape dynamically
        if x.dim() == 4:
            batch_size = x.size(0)
            
            # Case 1: Input is [batch_size, channels > self.input_channels, height, width]
            if x.size(1) > self.input_channels:
                # Use only the first self.input_channels channels
                x = x[:, :self.input_channels, :, :]
                
            # Case 2: Input is [batch_size, channels < self.input_channels, height, width]
            elif x.size(1) < self.input_channels:
                # Pad with zeros to match expected channels
                padding = torch.zeros(batch_size, self.input_channels - x.size(1), x.size(2), x.size(3), 
                                    device=x.device)
                x = torch.cat([x, padding], dim=1)
                
            # Ensure height dimension is 1
            if x.size(2) != 1:
                x = x.reshape(batch_size, self.input_channels, 1, -1)
        
        # CNN layers with ReLU activation and max pooling
        x = self.pool1(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool3(F.relu(self.conv3(x)))
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return F.softmax(x, dim=1)
    
    def reconfigure_for_channels(self, new_channels):
        """
        Method to reconfigure the model for a different number of input channels
        """
        # Ensure new_channels is within valid range (1 to 32)
        new_channels = max(1, min(new_channels, 32))
        
        if new_channels != self.input_channels:
            # Update input channels
            self.input_channels = new_channels
            
            # Recreate first convolutional layer with new input channels
            self.conv1 = nn.Conv2d(self.input_channels, 3, kernel_size=(1, 7), padding=(0, 3))
            
            # Recalculate flat size
            self._calculate_flat_size()
            
            # Recreate first fully connected layer with new flat size
            self.fc1 = nn.Linear(self.flat_size, 14)
            
            print(f"Model reconfigured for {self.input_channels} input channels")


class DSNNSystem:
    def __init__(self, model):
        self.model = model
        
    def shift_sequence(self, sequence, shift_right=True):
        """
        Shifts the input sequence one step right or left
        sequence: tensor of shape (batch_size, channels, 1, sequence_length)
        """
        if shift_right:
            return torch.roll(sequence, shifts=1, dims=-1)
        return torch.roll(sequence, shifts=-1, dims=-1)
    
    def voter(self, original_pred, shifted_pred):
        """
        Implements voting mechanism to combine predictions
        from original and shifted sequences
        """
        combined_pred = (original_pred + shifted_pred) / 2
        return torch.argmax(combined_pred, dim=1)
    
    def process_ecg(self, ecg_sequence):
        """
        Process a single ECG sequence through the DSNN system
        """
        # Ensure model is in evaluation mode
        self.model.eval()
        
        with torch.no_grad():
            # Ensure input channels match model's expected channels
            if ecg_sequence.size(1) != self.model.input_channels:
                # If channels don't match, dynamically reconfigure or process input
                if hasattr(self.model, 'reconfigure_for_channels'):
                    self.model.reconfigure_for_channels(ecg_sequence.size(1))
            
            # Original sequence prediction
            original_pred = self.model(ecg_sequence)
            
            # Shifted sequence prediction
            shifted_sequence = self.shift_sequence(ecg_sequence)
            shifted_pred = self.model(shifted_sequence)
            
            # Get final prediction through voting
            final_prediction = self.voter(original_pred, shifted_pred)
            
        return final_prediction

# Create sample ECG data with variable channels
def create_sample_ecg_data(num_samples=100, num_channels=2):
    """Generate synthetic ECG-like data for testing with variable channels"""
    # Create time points
    t = np.linspace(0, 23, 24)
    
    # Generate sample data for all channels
    samples = []
    labels = []
    
    for i in range(num_samples):
        # Create synthetic ECG-like signals for each channel
        channels = []
        for c in range(num_channels):
            # Generate slightly different signals for each channel
            signal = np.sin(t + np.random.rand() + c*0.5) + np.random.normal(0, 0.1, 24)
            channels.append(signal)
        
        # Stack all channels
        sample = np.stack(channels)
        samples.append(sample)
        
        # Generate random labels (0-5 for 6 classes)
        labels.append(np.random.randint(0, 6))
    
    return np.array(samples), np.array(labels)
