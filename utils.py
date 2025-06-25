"""
Training Utilities
------------------
Includes:
- Model training loop
- Checkpoint saving/loading
- Learning rate monitoring
"""

import torch
from PIL import Image

# Configuration
MODEL_CHECKPOINT_PATH = './models/weights/model.pth'
LOG_FILE_PATH = './models/weights/log.txt'

def trainModel(epochs, model, optimizer, scheduler, loss_fn, train_loader, test_loader, device):
    """Main training loop with checkpoint resuming"""
    # Resume training if checkpoint exists
    start_epoch, best_loss = loadModel(MODEL_CHECKPOINT_PATH)

    for epoch in range(start_epoch, epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_idx, (train_sketch, train_original) in enumerate(train_loader):
            train_sketch = train_sketch.to(device)
            train_original = train_original.to(device)
            
            optimizer.zero_grad()
            train_output = model(train_sketch)
            loss = loss_fn(train_output, train_original)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_idx, (test_sketch, test_original) in enumerate(test_loader):
                test_sketch = test_sketch.to(device)
                test_original = test_original.to(device)
                test_output = model(test_sketch)
                loss = loss_fn(test_output, test_original)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
        scheduler.step(avg_train_loss)  # Update learning rate

        # Save best model
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            saveModel(model, optimizer, scheduler, epoch, best_loss)
        
        # Logging
        with open(LOG_FILE_PATH, "a") as log_file:
            log_file.write(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}\n")

def loadModel(model_path):
    """Load model checkpoint if available"""
    try:
        checkpoint = torch.load(MODEL_CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']
        print(f"Resuming training from epoch {start_epoch}")
    except FileNotFoundError:
        print("No checkpoint found, starting fresh.")
        start_epoch = 0
        best_loss = float('inf')
    return start_epoch, best_loss
    
def saveModel(model, optimizer, scheduler, epoch, best_loss):
    """Save training checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': best_loss},
        MODEL_CHECKPOINT_PATH
    )