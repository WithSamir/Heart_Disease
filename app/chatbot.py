"""
chatbot.py — AI Health Chatbot module
Uses OpenAI if OPENAI_API_KEY is set, else falls back to a rule-based Q&A engine.
"""

import os
import re

# ── Rule-based Knowledge Base ─────────────────────────────────────────────────
KNOWLEDGE_BASE = {
    r"cholesterol": (
        "🔬 **Cholesterol** is a fatty substance in your blood. "
        "Normal: <200 mg/dL | Borderline: 200–239 | High: ≥240.\n"
        "Tip: Eat more fibre, oats, nuts; reduce saturated fats."
    ),
    r"blood pressure|bp": (
        "💉 **Blood Pressure** is written as systolic/diastolic (e.g., 120/80 mmHg).\n"
        "Normal: <120/80 | Elevated: 120–129 | High Stage 1: 130–139 | Stage 2: ≥140.\n"
        "Tip: Reduce salt, exercise regularly, manage stress."
    ),
    r"heart rate|pulse": (
        "❤️ **Resting Heart Rate**: Normal range is 60–100 bpm.\n"
        "Athletes may have 40–60 bpm. >100 bpm may indicate tachycardia.\n"
        "Tip: Aerobic exercise lowers resting heart rate over time."
    ),
    r"stress": (
        "😓 **Chronic stress** raises cortisol, which increases heart rate and BP.\n"
        "Tip: Try meditation, yoga, deep breathing, or walking 30 min/day."
    ),
    r"exercise|workout|physical activity": (
        "🏃 **Exercise** is one of the most powerful heart protectors.\n"
        "Aim for 150 min/week of moderate aerobic activity (brisk walking, cycling).\n"
        "Even 30 min/day significantly reduces heart disease risk."
    ),
    r"smoking|cigarette|tobacco": (
        "🚬 **Smoking** damages blood vessels and greatly increases heart disease risk.\n"
        "Quitting smoking halves your risk within 1 year.\n"
        "Resources: nicotineanonymous.org | iQuitline (India): 1800-11-2356"
    ),
    r"diabetes|blood sugar|glucose": (
        "🍬 **Diabetes** doubles heart disease risk by damaging blood vessels.\n"
        "Normal fasting glucose: <100 mg/dL | Prediabetes: 100–125 | Diabetes: ≥126.\n"
        "Tip: Low-glycemic diet, regular exercise, and medication adherence are key."
    ),
    r"alcohol|drink|drinking": (
        "🍺 **Alcohol** in excess raises BP and adds empty calories.\n"
        "Limit: ≤1 drink/day (women), ≤2 drinks/day (men).\n"
        "Heavy drinking is a significant risk factor for cardiomyopathy."
    ),
    r"obesity|bmi|overweight|weight": (
        "⚖️ **Obesity** (BMI ≥30) significantly raises heart disease risk.\n"
        "Even losing 5–10% of body weight improves heart health markers.\n"
        "Tip: Sustainable diet + 150 min/week exercise is the gold standard."
    ),
    r"chest pain|angina": (
        "⚠️ **Chest pain types:**\n"
        "• Typical Angina: Pressure/squeeze, worsens with exertion, relieves with rest\n"
        "• Atypical Angina: May not fit classic pattern\n"
        "• Non-Anginal: Not heart-related\n"
        "⛑️ **Seek emergency care immediately for sudden severe chest pain.**"
    ),
    r"family history|genetics|hereditary": (
        "🧬 **Family History** is a non-modifiable risk factor.\n"
        "If a first-degree relative (parent/sibling) had heart disease before 55 (men) or 65 (women), "
        "your risk is elevated.\n"
        "Tip: Earlier screening, lifestyle modification, and proactive monitoring are essential."
    ),
    r"risk|high risk|low risk|prediction|result": (
        "📊 **Understanding your risk score:**\n"
        "• 0–30% → Low Risk: Maintain healthy habits\n"
        "• 30–60% → Moderate Risk: Consult a doctor\n"
        "• 60–100% → High Risk: Seek medical attention soon\n"
        "🩺 Always verify with a qualified cardiologist — AI is not a substitute for medical advice."
    ),
    r"prevention|prevent|reduce risk": (
        "🛡️ **Heart Disease Prevention Checklist:**\n"
        "✅ No smoking\n✅ Exercise 150 min/week\n✅ Eat heart-healthy diet\n"
        "✅ Maintain healthy weight\n✅ Control BP, cholesterol, blood sugar\n"
        "✅ Manage stress\n✅ Limit alcohol\n✅ Regular health check-ups"
    ),
    r"diet|food|eat|nutrition": (
        "🥗 **Heart-Healthy Diet:**\n"
        "• 🟢 Eat more: fruits, vegetables, whole grains, fish, nuts, olive oil\n"
        "• 🔴 Limit: red meat, processed foods, trans fats, excess salt, sugar\n"
        "• Consider: Mediterranean or DASH diet — both proven to reduce heart risk"
    ),
    r"symptom|sign|warning": (
        "🚨 **Warning Signs of Heart Disease:**\n"
        "• Chest pain, pressure, or tightness\n"
        "• Shortness of breath\n"
        "• Pain radiating to arm, neck, or jaw\n"
        "• Unexplained fatigue\n"
        "• Swelling in legs/ankles\n"
        "⛑️ Call emergency services (112 in India) if experiencing these symptoms."
    ),
    r"hello|hi|hey|greet": (
        "👋 Hello! I'm **CardioBot**, your AI heart health assistant.\n"
        "Ask me about cholesterol, blood pressure, symptoms, diet, exercise, or your risk score!"
    ),
    r"thank|thanks": (
        "😊 You're welcome! Remember, your heart health is your greatest wealth. "
        "Stay active, eat well, and schedule regular check-ups! ❤️"
    ),
}

def rule_based_response(query: str) -> str:
    """Return a knowledge base response or a fallback."""
    q = query.lower()
    for pattern, answer in KNOWLEDGE_BASE.items():
        if re.search(pattern, q):
            return answer
    return (
        "🤔 I'm not sure about that specific topic. Here are things I can help with:\n"
        "• Cholesterol, Blood Pressure, Heart Rate\n"
        "• Symptoms, Prevention, Diet\n"
        "• Smoking, Stress, Diabetes, Obesity\n"
        "• Understanding your risk score\n\n"
        "Type any of these topics or ask a related question!"
    )


def get_chatbot_response(query: str, history: list | None = None) -> str:
    """
    Main entry point. Uses OpenAI GPT if API key is available, else rule-based.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are CardioBot, a friendly and knowledgeable AI health assistant "
                        "specializing in heart disease prevention, lifestyle guidance, and "
                        "understanding medical metrics. You explain complex medical concepts in simple, "
                        "encouraging language. Always remind users that you are not a doctor and "
                        "to consult medical professionals for diagnosis. Keep responses under 150 words."
                    ),
                }
            ]

            # Add chat history (last 6 turns)
            if history:
                for turn in history[-6:]:
                    messages.append({"role": "user",      "content": turn["user"]})
                    messages.append({"role": "assistant", "content": turn["bot"]})

            messages.append({"role": "user", "content": query})

            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=250,
                temperature=0.7,
            )
            return resp.choices[0].message.content.strip()

        except Exception as e:
            return rule_based_response(query) + f"\n\n_(AI mode unavailable: {e})_"

    return rule_based_response(query)
