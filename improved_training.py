"""
Train for 30 epochs to get better results - NO JIGSAW
"""
import torch
from mini_poly_tf import *

def train_improved():
    print("="*60)
    print("IMPROVED TRAINING: 30 Epochs for Better Accuracy")
    print("="*60)
    
    # Load your previous model (optional - warm start)
    model = SimpleVisionPolyTF()
    try:
        model.load_state_dict(torch.load('my_poly_tf_cifar.pth'))
        print(" Loaded previous model for warm start")
    except:
        print("Starting from scratch")
    
    # Setup training with improvements
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(" Using GPU")
    else:
        device = torch.device('cpu')
        print(" Using CPU - GPU not found")
    model = model.to(device)
    
    train_dataset = MultiTaskCIFAR(split='train')
    test_dataset = MultiTaskCIFAR(split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
    
    # Better optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    criterion = torch.nn.CrossEntropyLoss()
    
    print(f"\n Training for 30 epochs on {device}...")
    
    for epoch in range(30):
        model.train()
        train_correct = {'class': 0, 'rot': 0}
        train_total = 0
        
        for batch in train_loader:
            images = batch['image'].to(device)
            rotated = batch['rotated'].to(device)
            
            out_class = model(images, 'class')
            out_rot = model(rotated, 'rot')
            
            loss = (criterion(out_class, batch['class_label'].to(device)) +
                   criterion(out_rot, batch['rot_label'].to(device)))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_total += images.size(0)
            train_correct['class'] += (out_class.argmax(1) == batch['class_label'].to(device)).sum().item()
            train_correct['rot'] += (out_rot.argmax(1) == batch['rot_label'].to(device)).sum().item()
        
        scheduler.step()
        
        # Validation every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            test_correct = {'class': 0, 'rot': 0}
            test_total = 0
            
            with torch.no_grad():
                for batch in test_loader:
                    images = batch['image'].to(device)
                    rotated = batch['rotated'].to(device)
                    
                    out_class = model(images, 'class')
                    out_rot = model(rotated, 'rot')
                    
                    test_total += images.size(0)
                    test_correct['class'] += (out_class.argmax(1) == batch['class_label'].to(device)).sum().item()
                    test_correct['rot'] += (out_rot.argmax(1) == batch['rot_label'].to(device)).sum().item()
            
            acc = {k: v/test_total for k, v in test_correct.items()}
            print(f"Epoch {epoch+1}: Class={acc['class']:.3f}, Rot={acc['rot']:.3f}")
    
    torch.save(model.state_dict(), 'my_poly_tf_cifar_improved.pth')
    print("\n Improved model saved!")
    
    # Show predictions AFTER training
    print("\n Sample predictions on test images:")
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)
    batch = next(iter(test_loader))
    images = batch['image'].to(device)
    labels = batch['class_label']
    
    with torch.no_grad():
        outputs = model(images, 'class')
        preds = outputs.argmax(1)
    
    class_names = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']
    for i in range(10):
        print(f"True: {class_names[labels[i]]}, Pred: {class_names[preds[i]]}")

if __name__ == "__main__":
    train_improved()