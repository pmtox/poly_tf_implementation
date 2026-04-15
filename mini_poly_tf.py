"""
VISION POLY-TF - Running on YOUR CIFAR-10 data!
Single model doing 2 tasks: Classification and Rotation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# ============================================
# CHECK YOUR CIFAR-10 DATA
# ============================================

def check_cifar_data():
    """Verify your CIFAR-10 files are ready"""
    cifar_path = './cifar-10-batches-py'
    
    if os.path.exists(cifar_path):
        files = os.listdir(cifar_path)
        data_files = [f for f in files if f.startswith('data_batch_')]
        print(f" Found CIFAR-10 at: {cifar_path}")
        print(f" Data batches: {len(data_files)} files")
        print(f" Test batch: {'test_batch' in files}")
        return True
    else:
        print(" CIFAR-10 not found in current directory")
        print("  Will download automatically...")
        return False


# ============================================
# MULTI-TASK CIFAR DATASET
# ============================================

class MultiTaskCIFAR(Dataset):
    """CIFAR-10 with 2 tasks for Poly-TF"""
    def __init__(self, split='train'):
        if split == 'train':
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                   (0.2023, 0.1994, 0.2010))
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                   (0.2023, 0.1994, 0.2010))
            ])
        
        self.cifar = torchvision.datasets.CIFAR10(
            root='.',
            train=(split=='train'),
            download=False,
            transform=transform
        )
        self.split = split
        
    def __len__(self):
        return len(self.cifar)
    
    def __getitem__(self, idx):
        img, class_label = self.cifar[idx]
        
        # Task 1: Classification
        task_class = class_label
        
        # Task 2: Rotation (0°, 90°, 180°, 270°)
        rot_angle = np.random.randint(0, 4)
        img_rotated = self.rotate_image(img, rot_angle)
        
        return {
            'image': img,
            'rotated': img_rotated,
            'class_label': torch.tensor(class_label),
            'rot_label': torch.tensor(rot_angle)
        }
    
    def rotate_image(self, img, angle_idx):
        if angle_idx == 0:
            return img
        elif angle_idx == 1:
            return torch.rot90(img, 1, [1, 2])
        elif angle_idx == 2:
            return torch.rot90(img, 2, [1, 2])
        else:
            return torch.rot90(img, 3, [1, 2])


# ============================================
# SIMPLIFIED VISION POLY-TF
# ============================================

class SimpleVisionPolyTF(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.prompts = nn.ParameterDict({
            'class': nn.Parameter(torch.randn(1, 128) * 0.01),
            'rot': nn.Parameter(torch.randn(1, 128) * 0.01)
        })
        
        self.class_head = nn.Linear(128, 10)
        self.rot_head = nn.Linear(128, 4)
        
    def forward(self, x, task):
        features = self.shared(x)
        features = features.view(features.size(0), -1)
        
        if task == 'class':
            features = features + self.prompts['class']
            return self.class_head(features)
        else:
            features = features + self.prompts['rot']
            return self.rot_head(features)


# ============================================
# TRAINING
# ============================================

def train_poly_tf():
    print("="*60)
    print("VISION POLY-TF with YOUR CIFAR-10 Data")
    print("="*60)
    
    check_cifar_data()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n Using device: {device}")
    
    print("\n Loading CIFAR-10 from your files...")
    train_dataset = MultiTaskCIFAR(split='train')
    test_dataset = MultiTaskCIFAR(split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    print(f" Train: {len(train_dataset)} images")
    print(f" Test: {len(test_dataset)} images")
    
    model = SimpleVisionPolyTF().to(device)
    print(f"\n Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    
    print("\n Training (10 epochs)...")
    print("   (Model learns classification + rotation simultaneously)\n")
    
    train_acc_history = {'class': [], 'rot': []}
    test_acc_history = {'class': [], 'rot': []}
    
    for epoch in range(10):
        model.train()
        train_correct = {'class': 0, 'rot': 0}
        train_total = 0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/10")
        for batch in progress:
            images = batch['image'].to(device)
            rotated = batch['rotated'].to(device)
            
            labels_class = batch['class_label'].to(device)
            labels_rot = batch['rot_label'].to(device)
            
            out_class = model(images, 'class')
            out_rot = model(rotated, 'rot')
            
            loss = criterion(out_class, labels_class) + criterion(out_rot, labels_rot)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_total += images.size(0)
            train_correct['class'] += (out_class.argmax(1) == labels_class).sum().item()
            train_correct['rot'] += (out_rot.argmax(1) == labels_rot).sum().item()
            
            progress.set_postfix({
                'loss': f'{loss.item():.3f}',
                'cls': f'{train_correct["class"]/train_total:.2f}',
                'rot': f'{train_correct["rot"]/train_total:.2f}'
            })
        
        model.eval()
        test_correct = {'class': 0, 'rot': 0}
        test_total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(device)
                rotated = batch['rotated'].to(device)
                
                labels_class = batch['class_label'].to(device)
                labels_rot = batch['rot_label'].to(device)
                
                out_class = model(images, 'class')
                out_rot = model(rotated, 'rot')
                
                test_total += images.size(0)
                test_correct['class'] += (out_class.argmax(1) == labels_class).sum().item()
                test_correct['rot'] += (out_rot.argmax(1) == labels_rot).sum().item()
        
        train_acc = {k: v/train_total for k, v in train_correct.items()}
        test_acc = {k: v/test_total for k, v in test_correct.items()}
        
        for k in train_acc_history:
            train_acc_history[k].append(train_acc[k])
            test_acc_history[k].append(test_acc[k])
        
        print(f"\n  Train: Class={train_acc['class']:.3f}, Rot={train_acc['rot']:.3f}")
        print(f"  Test:  Class={test_acc['class']:.3f}, Rot={test_acc['rot']:.3f}")
        
        scheduler.step()
    
    plot_results(train_acc_history, test_acc_history)
    analyze_prompts(model)
    
    torch.save(model.state_dict(), 'my_poly_tf_cifar.pth')
    print("\n Model saved to 'my_poly_tf_cifar.pth'")
    
    return model, test_acc


def plot_results(train_acc, test_acc):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    tasks = ['class', 'rot']
    task_names = ['Classification', 'Rotation']
    
    for idx, (task, name) in enumerate(zip(tasks, task_names)):
        axes[idx].plot(train_acc[task], label='Train', linewidth=2)
        axes[idx].plot(test_acc[task], label='Test', linewidth=2)
        axes[idx].set_xlabel('Epoch')
        axes[idx].set_ylabel('Accuracy')
        axes[idx].set_title(f'{name}')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_ylim(0, 1)
    
    plt.suptitle('Poly-TF: Single Model Learning 2 Tasks', fontsize=14)
    plt.tight_layout()
    plt.savefig('poly_tf_cifar_results.png', dpi=150)
    print("\n Saved results to 'poly_tf_cifar_results.png'")
    plt.show()


def analyze_prompts(model):
    print("\n" + "="*60)
    print(" TASK PROMPT ANALYSIS")
    print("="*60)
    
    prompts = {
        'Classification': model.prompts['class'].detach().cpu().numpy().flatten(),
        'Rotation': model.prompts['rot'].detach().cpu().numpy().flatten()
    }
    
    print("\nPrompt Similarity:")
    names = list(prompts.keys())
    corr = np.corrcoef(prompts[names[0]], prompts[names[1]])[0,1]
    print(f"  {names[0]} vs {names[1]}: {corr:.4f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    
    for idx, (name, prompt) in enumerate(prompts.items()):
        prompt_2d = prompt[:128].reshape(8, 16)
        im = axes[idx].imshow(prompt_2d, cmap='coolwarm', aspect='auto')
        axes[idx].set_title(f'{name} Prompt')
        axes[idx].axis('off')
        plt.colorbar(im, ax=axes[idx])
    
    plt.suptitle('Poly-TF: Different Prompts for Different Tasks', fontsize=14)
    plt.tight_layout()
    plt.savefig('poly_tf_prompts_analysis.png', dpi=150)
    print("\n Saved prompt visualization to 'poly_tf_prompts_analysis.png'")


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    model, final_acc = train_poly_tf()
    
    print("\n" + "="*60)
    print(" SUCCESS! Vision Poly-TF trained on CIFAR-10")
    print("="*60)
    print(f"\nFinal Test Accuracies:")
    print(f"  • Classification: {final_acc['class']*100:.1f}%")
    print(f"  • Rotation:       {final_acc['rot']*100:.1f}%")
    print(f"  • Average:        {(final_acc['class']+final_acc['rot'])/2*100:.1f}%")
    
    print("\n Files generated:")
    print("  • my_poly_tf_cifar.pth - Trained model")
    print("  • poly_tf_cifar_results.png - Training curves")
    print("  • poly_tf_prompts_analysis.png - Prompt visualization")