import streamlit as st
from PIL import Image
from bedrock_client import generate_variations
from ui_components import render_file_upload, render_generation_controls, display_images
from utils import create_zip_download, upload_to_s3

def main():
    st.title("Image Dataset Augmentation")
    
    # File upload and validation
    valid_files = render_file_upload()
    
    if valid_files:
        # Generation controls
        num_variations, prompt_text, negative_prompt, similarity, prompt_strength = render_generation_controls()
        
        st.info(f"📊 Dataset Summary: {len(valid_files)} valid images × {num_variations} variations = {len(valid_files) * num_variations} total generated images")


        
        if st.button("Generate Variations"):
            all_images = []
            
            for uploaded_file in valid_files:
                st.markdown("---")
                st.subheader(f"Original: {uploaded_file.name}")

                original_image = Image.open(uploaded_file).convert("RGB")
                st.image(original_image, width='stretch')
                
                # Store original image
                all_images.append((f"original_{uploaded_file.name}", original_image))

                with st.spinner(f"Generating variations for {uploaded_file.name}..."):
                    variations = generate_variations(
                        original_image, prompt_text, negative_prompt, 
                        similarity, prompt_strength, num_variations
                    )

                display_images(variations, num_variations)
                
                # Store variations
                for i, (seed, variation) in enumerate(variations):
                    filename = f"{uploaded_file.name.split('.')[0]}_variation_{i+1}_seed_{seed}.png"
                    all_images.append((filename, variation))

            st.success("Done generating variations ✅")
            
            # Create zip download and upload to S3
            if all_images:
                zip_data, zip_filename = create_zip_download(all_images)
                
                # Upload to S3
                with st.spinner("Uploading to S3..."):
                    try:
                        folder_name, uploaded_files = upload_to_s3(all_images)
                        st.success(f"✅ Uploaded {len(uploaded_files)} images to S3 folder: {folder_name}")
                    except Exception as e:
                        st.error(f"❌ S3 upload failed: {str(e)}")
                
                st.download_button(
                    label="📥 Download All Images (ZIP)",
                    data=zip_data,
                    file_name=zip_filename,
                    mime="application/zip"
                )

if __name__ == "__main__":
    main()
