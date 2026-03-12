import streamlit as st
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import pandas as pd
import base64
from groq import Groq
import os


st.set_page_config(
    page_title="Indian Clothing Classifier",
    page_icon="👗",
    layout="centered"
)

st.title("Indian Clothing Image Classifier")

st.write(
"This app predicts the category of Indian clothing using either a "
"deep learning model (EfficientNet) or a Vision LLM."
)

st.markdown(
"""
### Test Images

If you don't have an image, you can download sample test images here:

🔗 https://drive.google.com/drive/folders/1Di0b2wPs-KbediLzKNYBkIQrrqfJ-81n?usp=sharing
"""
)


model_choice = st.selectbox(
    "Select Prediction Model",
    ["EfficientNet (CNN)", "Llama Vision (LLM)"]
)


classes = [
'BLOUSE',
'DHOTI PANTS or SALWAR',
'DUPATTA',
'GOWNS',
'KURTA MENS',
'LEGGINGS',
'LEHENGA',
'MENS MOJARI',
'NEHRU JACKET',
'PALAZZO',
'PETTICOAT',
'SAREE',
'SHERWANIS',
'WOMEN KURTA',
'WOMEN MOJARI'
]

device = torch.device("cpu")


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])


MODEL_PATH = os.path.join("Model", "best_EfficientNet.pth")


@st.cache_resource
def load_model():

    model = torchvision.models.efficientnet_b0()

    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features,
        len(classes)
    )

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )

    model.eval()

    return model


uploaded_file = st.file_uploader(
    "Upload an image of Indian clothing",
    type=["jpg","jpeg","png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_column_width=True
    )

    if model_choice == "EfficientNet (CNN)":

        model = load_model()

        img = transform(image).unsqueeze(0)

        with torch.no_grad():

            outputs = model(img)

            probabilities = torch.nn.functional.softmax(
                outputs,
                dim=1
            )[0]

        top3_prob, top3_idx = torch.topk(probabilities, 3)

        st.subheader("Top Predictions")

        results = []

        for prob, idx in zip(top3_prob, top3_idx):

            results.append({
                "Class": classes[idx],
                "Confidence (%)": round(prob.item()*100,2)
            })

        df = pd.DataFrame(results)

        st.table(df)

        chart_data = pd.DataFrame({
            "Class": classes,
            "Probability": probabilities.numpy()
        })

        chart_data = chart_data.sort_values(
            "Probability",
            ascending=False
        )

        st.bar_chart(chart_data.set_index("Class"))


    if model_choice == "Llama Vision (LLM)":

        st.info("Using Llama Vision model via Groq API")

        image_bytes = uploaded_file.getvalue()

        encoded = base64.b64encode(
            image_bytes
        ).decode("utf-8")

        image_url = f"data:image/jpeg;base64,{encoded}"

        client = Groq(
            api_key=st.secrets["GROQ_API_KEY"]
        )

        prompt = """
You are an expert fashion garment classifier.

Your task is to classify the clothing item into ONE of the categories below:
Kurta mens
Women kurta
Men Mojari
Women Mojari
Blouse
Leggings
Sherwani
Dhothi Pants
Saree
Salwar
Petticoat
Palazzo
Nehru Jacket
Lehenga
Gowns
Dupatta

Follow this visual inspection process carefully:

Step 1: Identify garment location
- upper body
- lower body
- full body
- accessory

Step 2: If it is a lower-body garment check structure

Check if garment has TWO separate legs
- If yes → it is pants → likely Palazzo, Salwar, Leggings, or Dhothi Pants
- If no → it is skirt style → likely Petticoat or Lehenga

Step 3: Distinguish Palazzo vs Petticoat

Palazzo:
- wide loose pants
- two separate legs
- looks like a skirt but actually pants
- waistband usually elastic
- worn as outer garment

Petticoat:
- single skirt structure
- NO separate legs
- worn under saree
- plain fabric
- narrow waist with cone shape

Step 4: Distinguish Petticoat vs Lehenga

Lehenga:
- heavy embroidery
- bridal or festive
- very wide flare

Petticoat:
- plain inner skirt
- simple fabric

Final rule:
If garment has TWO legs → choose Palazzo
If garment is SINGLE skirt with plain fabric → choose Petticoat

Return ONLY ONE label from the category list.
Do not explain.
"""

        completion = client.chat.completions.create(

            model="meta-llama/llama-4-scout-17b-16e-instruct",

            messages=[
                {
                    "role":"user",
                    "content":[
                        {"type":"text","text":prompt},
                        {
                            "type":"image_url",
                            "image_url":{"url":image_url}
                        }
                    ]
                }
            ],

            temperature=0,
            top_p=0.1,
            max_completion_tokens=10
        )

        prediction = completion.choices[0].message.content.strip()

        st.subheader("LLM Prediction")

        st.success(prediction)
