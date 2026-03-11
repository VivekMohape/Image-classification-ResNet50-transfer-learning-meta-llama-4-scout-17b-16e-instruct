import streamlit as st
import torch
import torchvision
from torchvision import transforms
from PIL import Image

# class labels
classes = [
'BLOUSE','DHOTI PANTS or SALWAR','DUPATTA','GOWNS',
'KURTA MENS','LEGGINGS','LEHENGA','MENS MOJARI',
'NEHRU JACKET','PALAZZO','PETTICOAT','SAREE',
'SHERWANIS','WOMEN KURTA','WOMEN MOJARI'
]

device = torch.device("cpu")

# image preprocessing
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

# load EfficientNet model
def load_model():

    model = torchvision.models.efficientnet_b0()

    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features,
        15
    )

    model.load_state_dict(
        torch.load("models/best_EfficientNet.pth", map_location=device)
    )

    model.eval()

    return model

model = load_model()

st.title("Indian Clothing Classifier")

st.write("Upload an image of Indian clothing.")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = transform(image).unsqueeze(0)

    with torch.no_grad():

        outputs = model(img)

        probs = torch.nn.functional.softmax(outputs, dim=1)

        pred = torch.argmax(probs,1).item()

    st.subheader("Prediction")
    st.write(classes[pred])

    st.subheader("Confidence")

    confidence = probs[0][pred].item()

    st.write(f"{confidence*100:.2f}%")
