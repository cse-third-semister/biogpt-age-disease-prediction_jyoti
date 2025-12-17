
''' 
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load BioGPT
tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT")
model = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT")

def explain_prediction(age, disease, risk, probability, symptoms):
    """
    Generate an explanation using BioGPT
    """

    risk = "Low"
    if probability > 0.6:
        risk = "High"
    elif probability > 0.3:
        risk = "Medium"

    prompt = f"""
You are a medical decision support system.

Patient Age: {age}
Predicted Disease: {disease}
Risk Level: {risk}
Prediction Confidence: {probability:.2f}
Symptoms: {symptoms}

Explain why this disease is likely, considering age-related factors.
Also mention that this is not a medical diagnosis.
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    explanation = generated_text.replace(prompt, "").strip()

    return explanation



# if __name__ == "__main__":
#     text = explain_prediction(
#         age=70,
#         disease="Heart Disease",
#         probability=0.78,
#         symptoms="chest pain, fatigue"
#     )

#     print(text)
'''
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load BioGPT
tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT")
model = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT")

def explain_prediction(age, disease, risk, probability, symptoms):
    """
    Generate an explanation using BioGPT
    BioGPT EXPLAINS the decision, it does NOT compute risk
    """

    prompt = f"""
You are a medical decision support system.

>Patient Age: {age}
>Predicted Disease: {disease}
>Risk Level: {risk}
>Prediction Confidence: {probability:.2f}
>Symptoms: {symptoms}
>This disease is likely, considering age-related factors.
 

"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    explanation = generated_text.replace(prompt, "").strip()

    return explanation
