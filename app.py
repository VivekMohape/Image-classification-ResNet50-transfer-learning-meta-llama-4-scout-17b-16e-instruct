import streamlit as st
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import pandas as pd

st.set_page_config(page_title="Indian Clothing Classifier", layout="centered")


# CLASS LABELS
classes = [
'BLOUSE','DHOTI PANTS or SALWAR','DUPATTA','GOWNS',
'KURTA MENS','LEGGINGS','LEHENGA','MENS MOJARI',
'NEHRU JACKET','PALAZZO','PETTICOAT','SAREE',
'SHERWANIS','WOMEN KURTA','WOMEN MOJARI'
]

device = torch.device("cpu")


# IMAGE TRANSFORM


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])


@st.cache_resource
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


# UI HEADER


st.title("Indian Clothing Image Classifier")

st.write(
"This app classifies Indian clothing items using a deep learning model "
"trained on the IndoFashion dataset."
)


# IMAGE UPLOAD


uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg","jpeg","png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = transform(image).unsqueeze(0)


    # MODEL PREDICTION


    with torch.no_grad():

        outputs = model(img)

        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]


    # TOP 3 PREDICTIONS


    top3_prob, top3_idx = torch.topk(probabilities, 3)

    st.subheader("Top Predictions")

    results = []

    for prob, idx in zip(top3_prob, top3_idx):

        label = classes[idx]

        confidence = prob.item() * 100

        results.append({
            "Class": label,
            "Confidence (%)": round(confidence,2)
        })

    df = pd.DataFrame(results)

    st.table(df)


    # PROBABILITY BAR CHART


    st.subheader("Prediction Confidence")

    chart_data = pd.DataFrame({
        "Class": classes,
        "Probability": probabilities.numpy()
    })

    chart_data = chart_data.sort_values("Probability", ascending=False)

    st.bar_chart(chart_data.set_index("Class"))
