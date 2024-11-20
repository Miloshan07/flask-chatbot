from flask import Flask, request, jsonify
import pandas as pd
import random
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer, util
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# Load Sentence Transformer Model
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# Pricing and FAQ Responses (Add your data here)
pricing_info = {      "basic business website": (
        "Basic Business Website - 30,000+ LKR\n"
        "Includes:\n"
        "• 5-page responsive website\n"
        "• Mobile-friendly design\n"
        "• Contact form integration\n"
        "• Basic SEO optimization\n"
        "• 1 year free hosting"
    ),
    "professional business website": (
        "Professional Business Website - 50,000+ LKR\n"
        "Includes:\n"
        "• Up to 10 pages of custom content\n"
        "• Responsive and modern design\n"
        "• Advanced SEO optimization\n"
        "• Social media integration\n"
        "• 1 year free hosting and maintenance"
    ),
    "premium business website": (
        "Premium Business Website - 70,000+ LKR\n"
        "Includes:\n"
        "• Unlimited pages\n"
        "• Custom design and functionality\n"
        "• Advanced SEO and analytics integration\n"
        "• Content Management System (CMS) if needed\n"
        "• Lifetime Hosting Free"
    ),
    "starter e-commerce website": (
        "Starter E-commerce Website - 90,000+ LKR\n"
        "Includes:\n"
        "• Up to 100 product listings\n"
        "• Responsive design\n"
        "• Basic payment gateway integration\n"
        "• Simple inventory management\n"
        "• 1 year free hosting"
    ),
    "advanced e-commerce website": (
        "Advanced E-commerce Website - 125,000+ LKR\n"
        "Includes:\n"
        "• Unlimited product listings\n"
        "• Custom design and branding\n"
        "• Multiple payment gateway options\n"
        "• Advanced inventory and order management\n"
        "• Customer account functionality\n"
        "• 1 year free hosting and maintenance"
    ),
    "enterprise e-commerce solution": (
        "Enterprise E-commerce Solution - 175,000+ LKR\n"
        "Includes:\n"
        "• Fully customized e-commerce platform\n"
        "• Advanced features (e.g., wishlist, reviews, loyalty program)\n"
        "• Multi-currency and multi-language support\n"
        "• Integration with ERP or CRM systems\n"
        "• 2 years free hosting and premium support"
        )
}

faq_responses = {     "customization": "Yes, you can switch from the Basic to Premium Business Website plan later.",
    "seo": "Basic SEO includes meta descriptions, while Advanced SEO covers analytics and backlinks.",
    "maintenance": "After the first year, maintenance packages start from 10,000 LKR annually.",
    "scalability": "Yes, you can upgrade from Starter to Enterprise without a full rebuild.",
    "payment gateways": "Includes PayPal and Stripe, suitable for international transactions.",
    "multi-currency support": "Implemented with live exchange rates using APIs.",
    "cms compatibility": "Supports WordPress, Shopify, and WooCommerce.",
    "custom features": "AI recommendations and chatbots can be integrated at additional cost.",
    "performance": "Optimized using Google Lighthouse and caching techniques.",
    "support policy": "24/7 support with issue resolution and free updates for 2 years.",
    "refund policy": "30-day refund policy if requirements are not met.",
    "legal compliance": "Ensures GDPR compliance with consent banners.",
    "comparison": "Our pricing is competitive with discounts for large projects.",
    "technology stack": "We use React, Node.js, Django, and Laravel." 
    }



# Sample data
corpus = [
    ("What does the basic business website include?", "basic business website"),
    ("Can you tell me more about the professional business website package?", "professional business website"),
    ("How many pages are included in the premium website plan?", "premium business website"),
    ("What is the cost of the starter e-commerce website?", "starter e-commerce website"),
    ("What features come with the advanced e-commerce website?", "advanced e-commerce website"),
    ("Is there an enterprise solution available for large e-commerce businesses?", "enterprise e-commerce solution"),
    ("Can I upgrade from a basic to a premium plan?", "customization"),
    ("What kind of SEO is included in the business websites?", "seo"),
    ("What happens after the free hosting period ends?", "maintenance"),
    ("Can I scale my starter e-commerce website later on?", "scalability"),
    ("Which payment gateways are supported in your plans?", "payment gateways"),
    ("How does the multi-currency feature work?", "multi-currency support"),
    ("Which CMS platforms are compatible with your services?", "cms compatibility"),
    ("Can I add custom features like AI to my website?", "custom features"),
    ("Do you offer speed optimization for the websites?", "performance"),
    ("What support do you provide for e-commerce solutions?", "support policy"),
    ("Do you have a refund policy if I'm not satisfied?", "refund policy"),
    ("Do you ensure the website meets GDPR compliance?", "legal compliance"),
    ("How does your pricing compare to others?", "comparison"),
    ("What technology stack do you use?", "technology stack")
]

# Text Preprocessing
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Load DataFrame
df = pd.DataFrame(corpus, columns=["query", "category"])
df["query"] = df["query"].apply(preprocess_text)
X_train, X_test, y_train, y_test = train_test_split(df['query'], df['category'], test_size=0.2)

# Train Model
pipeline = Pipeline([('tfidf', TfidfVectorizer()), ('classifier', LogisticRegression(max_iter=200))])
pipeline.fit(X_train, y_train)

# Predict Function
def predict_response(query):
    query = preprocess_text(query)
    predicted_category = pipeline.predict([query])[0]
    if predicted_category in pricing_info:
        return pricing_info[predicted_category]
    for keyword, response in faq_responses.items():
        if keyword in query:
            return response
    return "I'm not entirely sure about that. Please try a different query."

# API Route
@app.route("/predict", methods=["POST"])
def predict():
    user_input = request.json.get("query", "")
    if not user_input:
        return jsonify({"error": "No input query provided"}), 400
    response = predict_response(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
