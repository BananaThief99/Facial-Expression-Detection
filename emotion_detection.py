import torch
import torchvision.transforms as transforms
from torch.nn.functional import softmax
from networks.DDAM import DDAMNet

# Define the emotion labels
EMOTIONS = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger']

# Define the preprocessing transforms
TRANSFORMS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def load_emotion_model(model_path):
    num_classes = 7
    if 'ferPlus' in model_path:
        num_classes = 8
    model = DDAMNet(num_classes, pretrained=False)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def predict_emotions(face_image, model):
    """
    Predict the emotion of a given face image using the provided model.
    
    Args:
        face_image (numpy.ndarray): Cropped face image (as a NumPy array).
        model (torch.nn.Module): Trained PyTorch emotion detection model.

    Returns:
        str: Predicted emotion label.
        list: List of confidence scores for each emotion.
    """
    # Apply preprocessing
    face_tensor = TRANSFORMS(face_image).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        output = model(face_tensor)

        if isinstance(output, tuple):
            output = output[0]  # Handle tuple outputs if necessary

        confidences = softmax(output, dim=1)[0]
        predicted_emotion = EMOTIONS[torch.argmax(confidences).item()]

    return predicted_emotion, confidences.tolist()