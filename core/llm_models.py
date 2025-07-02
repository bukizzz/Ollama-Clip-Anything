from typing import Dict, List
from PIL import Image
import io
import base64
import torch
import torchvision.models as models
import torchvision.transforms as transforms


def query_imagebind(audio_path: str, image: Image.Image) -> Dict[str, List[float]]:
    """
    Queries a multimodal LLM (like a hypothetical ImageBind-enabled Ollama model)
    to generate embeddings or analyze audio-image correlation.
    
    Note: A true ImageBind implementation would involve loading a dedicated ImageBind model.
    This placeholder demonstrates how a multimodal LLM could be used for similar tasks
    if it supports both audio and image inputs. For now, it will simulate embeddings
    and print a message about the multimodal query.
    """
    print(f"🧠 Querying multimodal LLM for audio-image analysis with audio from {audio_path} and image data.")
    
    # Convert PIL Image to base64 string
    # base64_image = image_to_base64(image) # Removed unused variable
    
    # In a real scenario, you would load the audio file and convert it to a suitable format
    # for the multimodal LLM, e.g., base64 or a direct stream if supported.
    # For this placeholder, we'll just acknowledge the audio input.
    
    # Example of how you might structure a multimodal prompt for Ollama if it supported audio directly:
    # messages = [
    #     {
    #         'role': 'user',
    #         'content': "Analyze the relationship between this audio and image.",
    #         'images': [base64_image],
    #         'audio': [base64_audio] # Hypothetical audio input
    #     }
    # ]
    
    # For now, we'll just simulate the embeddings and acknowledge the inputs.
    # If LLM_MODEL is a multimodal model, you could use ollama.chat here with image input.
    # For a true ImageBind, you'd use its specific API.
    
    # Simulate ImageBind embeddings
    # In a real implementation, these would be derived from the model's output
    simulated_audio_embedding = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    simulated_image_embedding = [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
    
    print("✨ Simulating ImageBind embeddings.")
    return {"audio_embedding": simulated_audio_embedding, "image_embedding": simulated_image_embedding}

# Function to get image embeddings using a pre-trained ResNet model
def query_image_embedding(image: Image.Image) -> List[float]:
    """Generates an embedding for a given PIL Image using a pre-trained ResNet model."""
    print("🖼️ Generating image embedding using ResNet...")
    # Load the pre-trained ResNet model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # Remove the final classification layer to get features
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval() # Set to evaluation mode

    # Define image transformations
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Preprocess the image and add a batch dimension
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    # Move the model and input to GPU if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model = model.to('cuda')

    with torch.no_grad():
        embedding = model(input_batch)

    # Flatten the embedding and convert to a list
    return embedding.flatten().tolist()

def image_to_base64(image: Image.Image) -> str:
    """Converts a PIL Image to a base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def base64_to_image(base64_string: str) -> Image.Image:
    """Converts a base64 string to a PIL Image."""
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))
