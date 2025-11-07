import boto3
import json
import random
import streamlit as st
from PIL import Image
import io
import base64
from utils import encode_image

def create_image_guardrail():
    """Create or get existing Bedrock guardrail for image generation."""
    bedrock = boto3.client('bedrock')
    
    try:
        response = bedrock.create_guardrail(
            name='ImageAugmentationGuardrail',
            description='Filters inappropriate content for industrial dataset augmentation',
            contentPolicyConfig={
                'filtersConfig': [
                    {'type': 'HATE', 'inputStrength': 'HIGH', 'outputStrength': 'HIGH'},
                    {'type': 'VIOLENCE', 'inputStrength': 'LOW', 'outputStrength': 'LOW'},
                    {'type': 'SEXUAL', 'inputStrength': 'HIGH', 'outputStrength': 'HIGH'},
                    {'type': 'MISCONDUCT', 'inputStrength': 'MEDIUM', 'outputStrength': 'MEDIUM'}
                ]
            },
            blockedInputMessagingConfig={
                'message': 'Input prompt contains inappropriate content for industrial dataset generation.'
            },
            blockedOutputsMessagingConfig={
                'message': 'Generated image was filtered to ensure appropriate industrial content.'
            }
        )
        return response['guardrailId'], response['version']
    except Exception as e:
        if 'already exists' in str(e):
            guardrails = bedrock.list_guardrails()
            for guardrail in guardrails['guardrails']:
                if guardrail['name'] == 'ImageAugmentationGuardrail':
                    return guardrail['id'], guardrail['version']
        return None, None

def generate_variations(image, prompt_text, negative_prompt, similarity, prompt_strength, num_variations=5):
    """Generate image variations using Bedrock Titan model."""
    bedrock = boto3.client('bedrock-runtime')
    encoded_image = encode_image(image)
    
    variations = []
    attempts = 0
    max_attempts = num_variations * 3
    
    while len(variations) < num_variations and attempts < max_attempts:
        attempts += 1
        seed = random.randint(0, 999999999)
        
        body = {
            "taskType": "IMAGE_VARIATION",
            "imageVariationParams": {
                "images": [encoded_image],
                "text": prompt_text,
                "negativeText": negative_prompt,
                "similarityStrength": similarity,
            },
            "imageGenerationConfig": {
                "numberOfImages": 1,
                "height": 512,
                "width": 512,
                "cfgScale": prompt_strength,
                "seed": seed,
            },
        }
        
        try:
            guardrail_id, guardrail_version = create_image_guardrail()
            
            if guardrail_id:
                response = bedrock.invoke_model(
                    modelId="amazon.titan-image-generator-v2:0",
                    body=json.dumps(body),
                    guardrailIdentifier=guardrail_id,
                    guardrailVersion=guardrail_version
                )
            else:
                response = bedrock.invoke_model(
                    modelId="amazon.titan-image-generator-v2:0",
                    body=json.dumps(body)
                )
            
            result = json.loads(response['body'].read())
            if "images" in result and result["images"]:
                img_b64 = result["images"][0]
                img_bytes = base64.b64decode(img_b64)
                variations.append((seed, Image.open(io.BytesIO(img_bytes))))
        except Exception as e:
            if not ("blocked" in str(e).lower() or "filtered" in str(e).lower()):
                st.error(f"❌ Error generating variation: {str(e)}")
                break

    return variations