"""import torch
import torch.nn as nn
from Classification_metrics import ClassificationMetrics
def train_dsnn(model, train_loader, val_loader=None, num_epochs=2000, learning_rate=0.001, device='cuda', class_names=None):
    #removed 3 quotes
    Training function for DSNN with validation and metrics tracking
    #removed 3 quotes

    # Print model info
    print(f"Model structure: {model}")
    print(f"Model expected input channels: {model.input_channels}")
    print(f"Model expected sequence length: {model.sequence_length}")
    
    # Inspect the first batch to understand the data shape
    for data, target in train_loader:
        print(f"Input data shape: {data.shape}")
        print(f"Target shape: {target.shape}")
        break
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    model.to(device)
    model.train()
    
    # Get a sample batch to determine input channels
    for data, _ in train_loader:
        print(f"Input data shape: {data.shape}")
        break

    # Get actual input channels from data
    actual_channels = data.shape[1]
    expected_channels = model.conv1.in_channels
    
    print(f"Data has {actual_channels} channels, model expects {expected_channels} channels")
        
    # Recalculate flat_size if actual channels are different
    if actual_channels != model.input_channels:
        print(f"Adjusting model for {actual_channels} input channels (was {model.input_channels})")
        model.input_channels = actual_channels
        model.conv1 = nn.Conv2d(actual_channels, 3, kernel_size=(1, 7), padding=(0, 3)).to(device)
        model._calculate_flat_size(actual_channels, 24)
        model.fc1 = nn.Linear(model.flat_size, 14).to(device)
    
    # For tracking best model
    best_val_f1 = 0
    best_model_state = None

    # Initialize metrics calculator for validation
    if val_loader:
        metrics_calc = ClassificationMetrics(num_classes=model.fc2.out_features)
        if class_names:
            metrics_calc.set_class_names(class_names)

    model.to(device)

    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
                    
            # Training phase
            model.train()
            train_loss = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                # Reshape data if needed
                if data.dim() == 4 and data.size(1) != model.input_channels:
                    data = data.permute(0, 2, 1, 3)
                    data = data.reshape(data.size(0), model.input_channels, 1, -1)
                data, target = data.to(device), target.to(device)

                # Generate shifted data
                shifted_data = torch.roll(data, shifts=1, dims=-1)

                # Forward pass with original and shifted data
                optimizer.zero_grad()
                output_original = model(data)
                output_shifted = model(shifted_data)

                # Combine losses from both original and shifted data
                loss = criterion(output_original, target) + criterion(output_shifted, target)
                train_loss += loss.item()

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                if batch_idx % 100 == 0:
                    print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

            # Validation phase
            if val_loader:
                model.eval()
                metrics_calc.reset()
                val_loss = 0

                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        val_loss += criterion(output, target).item()

                        # Get predictions
                        _, predicted = torch.max(output, 1)

                        # Update metrics
                        metrics_calc.update(target.cpu().numpy(), predicted.cpu().numpy(), output.cpu().numpy())

                # Calculate validation metrics
                metrics = metrics_calc.get_metrics()
                val_f1 = metrics['macro_f1']

                print(f'Epoch: {epoch}, Train Loss: {train_loss/len(train_loader):.4f}, '
                      f'Val Loss: {val_loss/len(val_loader):.4f}, Val F1: {val_f1:.4f}')

                # Save best model
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_model_state = model.state_dict().copy()
                    print(f'New best model saved with validation F1: {best_val_f1:.4f}')

        # Load best model if validation was used
        if val_loader and best_model_state:
            model.load_state_dict(best_model_state)
            print(f'Loaded best model with validation F1: {best_val_f1:.4f}')

    return model"""
import torch
import torch.nn as nn
from Classification_metrics import ClassificationMetrics

def train_dsnn(model, train_loader, val_loader=None, num_epochs=2000, learning_rate=0.001, device='cuda', class_names=None):
    """
    Training function for DSNN with validation and metrics tracking.
    Supports dynamic adaptation to different channel counts.
    """
    # Move to specified device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Print model info
    print(f"Model structure: {model}")
    print(f"Model expected input channels: {model.input_channels}")
    print(f"Model expected sequence length: {model.sequence_length}")
    
    # Inspect the first batch to understand the data shape
    for data, target in train_loader:
        print(f"Input data shape: {data.shape}")
        print(f"Target shape: {target.shape}")
        break
    
    # Initialize loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Get actual input channels from data
    batch_size, actual_channels = data.shape[0], data.shape[1]
    
    print(f"Data has {actual_channels} channels, model expects {model.input_channels} channels")
    
    # Check if we need to adapt the model to the actual channels in the data
    if actual_channels != model.input_channels:
        print(f"Adapting model for {actual_channels} input channels (was {model.input_channels})")
        
        # Use the reconfigure_for_channels method if available
        if hasattr(model, 'reconfigure_for_channels'):
            model.reconfigure_for_channels(actual_channels)
        else:
            # Manual reconfiguration as fallback
            model.input_channels = actual_channels
            model.conv1 = nn.Conv2d(actual_channels, 3, kernel_size=(1, 7), padding=(0, 3))
            model._calculate_flat_size()
            model.fc1 = nn.Linear(model.flat_size, 14)
    
    # Move model to device after reconfiguration
    model.to(device)
    
    # For tracking best model
    best_val_f1 = 0
    best_model_state = None

    # Initialize metrics calculator for validation
    if val_loader:
        metrics_calc = ClassificationMetrics(num_classes=model.fc2.out_features)
        if class_names:
            metrics_calc.set_class_names(class_names)

    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        batch_count = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            batch_count += 1
            
            # Handle channel dimension properly
            # Check if reshaping is needed
            if data.dim() == 4:
                # If input data has a different channel count than the model expects
                if data.size(1) != model.input_channels:
                    if data.size(1) > model.input_channels:
                        # If data has more channels than model expects, use only what's needed
                        data = data[:, :model.input_channels, :, :]
                    elif data.size(1) < model.input_channels:
                        # If data has fewer channels than model expects, pad with zeros
                        padding = torch.zeros(data.size(0), model.input_channels - data.size(1), 
                                             data.size(2), data.size(3))
                        data = torch.cat([data, padding], dim=1)
                
                # Ensure height dimension is 1
                if data.size(2) != 1:
                    data = data.reshape(data.size(0), data.size(1), 1, -1)
                    
            # Move to device
            data, target = data.to(device), target.to(device)

            # Generate shifted data for the voting mechanism
            shifted_data = torch.roll(data, shifts=1, dims=-1)

            # Forward pass with original and shifted data
            optimizer.zero_grad()
            output_original = model(data)
            output_shifted = model(shifted_data)

            # Combine losses from both original and shifted data
            loss = criterion(output_original, target) + criterion(output_shifted, target)
            train_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

        # Calculate average training loss
        avg_train_loss = train_loss / batch_count
        print(f'Epoch: {epoch}, Average Train Loss: {avg_train_loss:.4f}')

        # Validation phase
        if val_loader:
            model.eval()
            metrics_calc.reset()
            val_loss = 0
            val_count = 0

            with torch.no_grad():
                for data, target in val_loader:
                    val_count += 1
                    
                    # Handle channel dimension the same way as in training
                    if data.dim() == 4:
                        if data.size(1) != model.input_channels:
                            if data.size(1) > model.input_channels:
                                data = data[:, :model.input_channels, :, :]
                            elif data.size(1) < model.input_channels:
                                padding = torch.zeros(data.size(0), model.input_channels - data.size(1), 
                                                     data.size(2), data.size(3))
                                data = torch.cat([data, padding], dim=1)
                        
                        if data.size(2) != 1:
                            data = data.reshape(data.size(0), data.size(1), 1, -1)
                    
                    # Move to device
                    data, target = data.to(device), target.to(device)
                    
                    # Forward pass
                    output = model(data)
                    val_loss += criterion(output, target).item()

                    # Get predictions
                    _, predicted = torch.max(output, 1)

                    # Update metrics
                    metrics_calc.update(target.cpu().numpy(), predicted.cpu().numpy(), output.cpu().numpy())

            # Calculate validation metrics
            metrics = metrics_calc.get_metrics()
            val_f1 = metrics['macro_f1']
            avg_val_loss = val_loss / val_count

            print(f'Epoch: {epoch}, Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}, Val F1: {val_f1:.4f}')

            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = model.state_dict().copy()
                print(f'New best model saved with validation F1: {best_val_f1:.4f}')

    # Load best model if validation was used
    if val_loader and best_model_state:
        model.load_state_dict(best_model_state)
        print(f'Loaded best model with validation F1: {best_val_f1:.4f}')

    return model
