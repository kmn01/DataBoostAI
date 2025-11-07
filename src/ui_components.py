import streamlit as st
from PIL import Image
from utils import validate_image

def render_file_upload():
    """Render file upload component with validation."""
    uploaded_files = st.file_uploader(
        "Upload images for augmentation (min: 256px, max: 4096px per dimension)", 
        type=['png', 'jpg', 'jpeg'], 
        accept_multiple_files=True
    )
    
    valid_files = []
    validation_results = []
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            is_valid, message = validate_image(uploaded_file)
            status = "✅" if is_valid else "❌"
            validation_results.append(f"{status} {uploaded_file.name}: {message}")
            
            if is_valid:
                valid_files.append(uploaded_file)
        
        # Show validation summary
        valid_count = len(valid_files)
        total_count = len(uploaded_files)
        
        if valid_count == total_count:
            st.success(f"✅ All {total_count} images are valid")
        else:
            st.warning(f"⚠️ {valid_count}/{total_count} images are valid")
        
        # Validation details in expander
        with st.expander("📋 View validation details"):
            for result in validation_results:
                st.write(result)
    
    return valid_files

def render_generation_controls():
    """Render generation parameter controls."""
    num_variations = st.slider("Number of variations per image", 1, 20, 5)
    
    prompt_text = st.text_area(
        "Prompt (variation instructions)",
        "Generate a realistic variation of the reference industrial scene. "
        "Keep the same overall style and professionalism, but slightly change lighting, "
        "PPE colors (helmets/vests), worker pose, machinery arrangement, and camera angle. "
        "Maintain correct PPE usage and an authentic industrial environment."
    )

    negative_prompt = st.text_area(
        "Negative Prompt",
        "cartoon, illustration, distorted proportions, duplicate people, extra limbs, "
        "unsafe behavior, missing PPE, blurry, text, watermark, logo, unrealistic lighting, "
        "unnatural colors, deformed hands, deformed faces"
    )

    similarity = st.slider(
        "Similarity — lower = more change, higher = closer to original",
        min_value=0.0, max_value=1.0, value=0.4, step=0.05
    )

    prompt_strength = st.slider(
        "Prompt Strength — lower = takes more liberties, higher = follows prompt more closely",
        min_value=1.0, max_value=10.0, value=8.0, step=0.5
    )
    
    return num_variations, prompt_text, negative_prompt, similarity, prompt_strength

def display_images(variations, num_variations):
    """Display generated variations in columns."""
    st.subheader("Generated Variations")
    cols = st.columns(num_variations)
    
    for i, (seed, variation) in enumerate(variations):
        with cols[i]:
            st.image(variation, width='stretch')