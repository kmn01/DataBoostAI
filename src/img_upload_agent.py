import streamlit as st
import boto3
import json
import base64
from PIL import Image
import io

def encode_image(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()

def generate_variations(image, num_variations=3):
    bedrock = boto3.client('bedrock-runtime')
    
    encoded_image = encode_image(image)
    
    prompts = [
        "different lighting conditions, bright daylight",
        "different background, outdoor setting", 
        "different colors, vibrant color scheme",
        "different angle, side view perspective",
        "different weather, rainy or cloudy conditions"
    ]
    
    variations = []
    for i in range(num_variations):
        body = json.dumps({
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {
                "text": f"Create a variation of this image with {prompts[i % len(prompts)]}, maintain the main subject and composition",
                "conditionImage": encoded_image,
                "controlStrength": 0.7
            },
            "imageGenerationConfig": {
                "numberOfImages": 1,
                "height": 512,
                "width": 512,
                "cfgScale": 8.0
            }
        })
        
        response = bedrock.invoke_model(
            modelId='amazon.titan-image-generator-v2:0',
            body=body
        )
        
        result = json.loads(response['body'].read())
        generated_image = base64.b64decode(result['images'][0])
        variations.append(Image.open(io.BytesIO(generated_image)))
    
    return variations

st.title("Image Dataset Augmentation")

uploaded_files = st.file_uploader(
    "Upload images for augmentation", 
    type=['png', 'jpg', 'jpeg'], 
    accept_multiple_files=True
)

if uploaded_files:
    num_variations = st.slider("Number of variations per image", 1, 5, 3)
    
    if st.button("Generate Variations"):
        for uploaded_file in uploaded_files:
            st.subheader(f"Original: {uploaded_file.name}")
            original_image = Image.open(uploaded_file)
            st.image(original_image, width='stretch')
            
            with st.spinner(f"Generating variations for {uploaded_file.name}..."):
                variations = generate_variations(original_image, num_variations)
                
                st.subheader("Generated Variations:")
                cols = st.columns(num_variations)
                for i, variation in enumerate(variations):
                    with cols[i]:
                        st.image(variation, caption=f"Variation {i+1}", width='stretch')