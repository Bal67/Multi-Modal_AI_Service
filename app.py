import os
import json
import time
import boto3
import torch
import streamlit as st
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
from io import BytesIO
from dotenv import load_dotenv
import urllib.request

# Load environment variables from .env file
load_dotenv()

# Ensure AWS credentials are available via environment or .env
region = os.getenv("AWS_REGION", "us-east-1")
bedrock_model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

# Initialize the Bedrock client
bedrock_client = boto3.client("bedrock-runtime", region_name=region)

# Load the pre-trained ResNet-50 model
resnet_model = models.resnet50(pretrained=True)
resnet_model.eval()

# Load ImageNet class labels from GitHub
imagenet_labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
imagenet_labels = urllib.request.urlopen(imagenet_labels_url).read().decode("utf-8").splitlines()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Analyze image locally with timing
def analyze_image_locally(image_file):
    start = time.time()
    image = Image.open(image_file).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = resnet_model(image_tensor)
    predicted_class = output.argmax(dim=1).item()
    label = imagenet_labels[predicted_class]
    duration = time.time() - start
    print(f"[PERF] Image processed in {duration:.3f} seconds")
    return label

# Analyze text via Claude + Bedrock with timing
def analyze_text_claude(text, context=""):
    start = time.time()
    messages = [{"role": "user", "content": f"{text}\n\nContext: {context}"}]
    body = json.dumps({
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0.7,
        "anthropic_version": "bedrock-2023-05-31"
    })
    response = bedrock_client.invoke_model(
        modelId=bedrock_model_id,
        body=body,
        contentType="application/json",
        accept="application/json"
    )
    result = json.loads(response["body"].read())
    duration = time.time() - start
    print(f"[PERF] Claude responded in {duration:.3f} seconds")
    return result.get("content", "No response")

# Streamlit app setup
st.set_page_config(page_title="Claude + ResNet", layout="wide")
st.title("Claude 3 + ResNet Multi-Modal Analyzer")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
user_input = st.chat_input("Ask something...")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Process user input and uploaded image
if user_input or uploaded_image:
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        response = analyze_text_claude(user_input)
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "assistant", "content": response})

    if uploaded_image:
        with st.chat_message("user"):
            st.markdown("Uploaded image")
        image_result = analyze_image_locally(uploaded_image)
        with st.chat_message("assistant"):
            st.markdown(f"ResNet prediction: **{image_result}**")
        st.session_state.chat_history.append({"role": "user", "content": "Uploaded image"})
        st.session_state.chat_history.append({"role": "assistant", "content": image_result})
