import torch
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr
from mini_poly_tf import SimpleVisionPolyTF

# Load your trained model
model = SimpleVisionPolyTF()
model.load_state_dict(torch.load('my_poly_tf_cifar_improved.pth', map_location='cpu'))
model.eval()

# Image preprocessing - MATCHES TRAINING
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
rot_names = ['0°', '90°', '180°', '270°']

def predict(image):
    img = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        class_out = model(img, 'class')
        rot_out = model(img, 'rot')
    
    class_pred = class_names[class_out.argmax().item()]
    rot_pred = rot_names[rot_out.argmax().item()]
    
    class_conf = torch.softmax(class_out, dim=1).max().item()
    rot_conf = torch.softmax(rot_out, dim=1).max().item()
    
    return {
        "Object Type": f"{class_pred} ({class_conf*100:.1f}%)",
        "Rotation": f"{rot_pred} ({rot_conf*100:.1f}%)"
    }

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.JSON(label="Predictions"),
    title="Poly-TF: Multi-Task Vision Model",
    description="One model doing two tasks: Classification and Rotation detection on CIFAR-10",
)

iface.launch()