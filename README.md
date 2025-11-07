# SafeVisionAI: Synthetic Data Platform for Industrial Safety

### Authors
- **Jiwon Bae** — MS Data Science @ Columbia University  
- **Blanca Valera Caballero** — MS Data Science @ Columbia University  
- **Kirthana Natarajan** — MS Computer Science @ Columbia University  

---

## 🚧 Overview
**SafeVisionAI** is a **Synthetic Data as a Service (SDaaS)** platform that generates **photorealistic industrial safety datasets** using **Amazon Bedrock**.  
It enables companies to train and improve **Computer Vision models for PPE compliance** (helmets, vests, goggles) without relying on expensive, privacy-sensitive real-world footage.

By transforming real seed images into diverse, realistic variations, SafeVisionAI:
- Cuts labeling costs by **80–90%**  
- Improves downstream model performance (**Δ F1 +6–12 points**)  
- Eliminates **PII and legal risk** from real footage  

---

## 💡 Problem & Solution

### Problem
Industrial AI systems struggle with:
- High labeling cost and long data collection cycles  
- Legal risks due to real employee footage  
- Fragile models that fail under poor lighting, odd angles, or non-compliance scenarios  

### Solution
SafeVisionAI uses **Bedrock’s Titan Image Generator v2** to create synthetic PPE scenes that:
- Cover rare “edge cases” safely and cheaply  
- Provide automatic annotations (based on structured prompts)  
- Scale instantly via AWS serverless architecture  

---

## ⚙️ Technical Architecture

| Component | Function | AWS Service |
|------------|-----------|--------------|
| **Frontend** | Image upload & generation config | Streamlit + API Gateway |
| **Generator Engine** | Image-to-image synthetic generation | Amazon Bedrock (Titan v2) |
| **Storage** | Dataset management & delivery | Amazon S3 |
| **Orchestration** | Prompt generation & quality verification | AWS Lambda + Step Functions |
| **Security & Guardrails** | Content safety & compliance filters | Bedrock Guardrails + VPC mode |

**Key Innovation:**  
Each synthetic image is generated *programmatically* — ensuring every sample is automatically labeled, reproducible (via seeds), and privacy-safe.

---

## 🧠 Model Implementation

We use **Amazon Titan Image Generator v2 (Bedrock)** in `IMAGE_VARIATION` mode with:
- A **reference image** (uploaded PPE scene)  
- **Instruction-style prompt**: defines what to vary (lighting, angle, PPE color)  
- **Negative prompt**: filters out unsafe or unrealistic content  
- **Parameters**:  
  - `controlStrength` = similarity (0.3–0.7)  
  - `cfgScale` = prompt strength (6–10)  
  - `seed` = controls reproducibility (randomized per run)

### Example Prompt
> Generate a realistic variation of the reference industrial scene.  
> Slightly change lighting and background machinery, modify PPE color, and ensure proper safety compliance.  
> Maintain realism and worker posture.

### Negative Prompt
> cartoon, illustration, unsafe behavior, distorted proportions, missing PPE, text, watermark, logo.

### Guardrails
SafeVisionAI integrates **Amazon Bedrock Guardrails** to:
- Block unsafe or biased generations  
- Enforce PPE and safety compliance  
- Ensure diversity in workers’ appearance and environment  

---

## 💼 Business Value

| KPI | Before | With SafeVisionAI |
|------|---------|-------------------|
| Labeling Cost (per 1K images) | ~$700 | ~$40 |
| Model F1 (example) | 78% | 88–90% |
| Dataset Delivery | Weeks | Hours |
| Legal Risk | High | None (synthetic) |

**ROI:** For 10K images/year → payback within **1–3 months**.

---

## 🔮 Future Work
- Verifier Agents: Auto-filter low-quality or unsafe generations.
- Auto-Annotation: Export bounding boxes / YOLO labels.
- Bias & Diversity Dashboard: Measure demographic and angle coverage.
- API & Marketplace Integration: Deliver SDaaS via Bedrock pipelines.
