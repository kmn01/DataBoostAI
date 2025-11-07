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

def generate_variations(
    image: Image.Image,
    num_variations=5,
    prompt_text: str,
    negative_prompt: str,
    similarity: float,
    prompt_strength: float,
    base_seed: int,
):
    bedrock = boto3.client('bedrock-runtime')
    
    encoded_image = encode_image(image)
    
    variations = []
    for i in range(num_variations):
        seed = int(base_seed + i)
        
        body = {
            "taskType": "IMAGE_VARIATION",
            "textToImageParams": {
                "text": prompt_text,
                "negativeText": negative_prompt,
                "conditionImage": encoded_image,
                "controlStrength": similarity,  # similarity / how much to follow base
            },
            "imageGenerationConfig": {
                "numberOfImages": 1,
                "height": 512,
                "width": 512,
                "cfgScale": prompt_strength,   # prompt strength
                "seed": seed,
            },
        }
        
        response = bedrock.invoke_model(
            modelId="amazon.titan-image-generator-v2:0",
            body=json.dumps(body),
        )
        
        result = json.loads(response['body'].read())
        img_b64 = result["images"][0]
        img_bytes = base64.b64decode(img_b64)
        variations.append((seed, Image.open(io.BytesIO(img_bytes))))

    return variations

st.title("Image Dataset Augmentation")

uploaded_files = st.file_uploader(
    "Upload images for augmentation", 
    type=['png', 'jpg', 'jpeg'], 
    accept_multiple_files=True
)

if uploaded_files:
    # num_variations = st.slider("Number of variations per image", 1, 5, 3)

    prompt_text = st.text_area(
        "Prompt (variation instructions)",
        (
            "Generate a realistic variation of the reference industrial scene. "
            "Keep the same overall style and professionalism, but slightly change lighting, "
            "PPE colors (helmets/vests), worker pose, machinery arrangement, and camera angle. "
            "Maintain correct PPE usage and an authentic industrial environment."
        ),
    )

    negative_prompt = st.text_area(
        "Negative Prompt",
        (
            "cartoon, illustration, distorted proportions, duplicate people, extra limbs, "
            "unsafe behavior, missing PPE, blurry, text, watermark, logo, unrealistic lighting, "
            "unnatural colors, deformed hands, deformed faces"
        ),
    )

    similarity = st.slider(
        "Similarity (controlStrength) — lower = more change, higher = closer to original",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.05,
    )

    prompt_strength = st.slider(
        "Prompt Strength (cfgScale)",
        min_value=1.0,
        max_value=20.0,
        value=8.0,
        step=0.5,
    )

    base_seed = st.number_input(
        "Base Seed (different seeds → different variations)",
        min_value=0,
        value=0,
        step=1,
    )
    
    if st.button("Generate Variations"):
        for uploaded_file in uploaded_files:
            st.markdown("---")
            st.subheader(f"Original: {uploaded_file.name}")

            original_image = Image.open(uploaded_file).convert("RGB")
            st.image(original_image, use_column_width=True)

            with st.spinner(f"Generating variations for {uploaded_file.name}..."):
                variations = generate_variations(
                    image=original_image,
                    num_variations=num_variations,
                    prompt_text=prompt_text,
                    negative_prompt=negative_prompt,
                    similarity=similarity,
                    prompt_strength=prompt_strength,
                    base_seed=base_seed,
                )

            st.subheader("Generated Variations")
            cols = st.columns(num_variations)

            for i, (seed, variation) in enumerate(variations):
                with cols[i]:
                    st.image(
                        variation,
                        caption=(
                            f"Variation {i+1}\n"
                            f"Model: Titan v2 | Seed: {seed}\n"
                            f"Similarity: {similarity} | Prompt Strength: {prompt_strength}"
                        ),
                        use_column_width=True,
                    )

        st.success("Done generating variations ✅")
