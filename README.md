# 🤖 AI Customer Support Copilot — ZENDS Communications

An AI-powered customer support copilot designed to help telecom support agents understand customer queries, identify intent and sentiment, retrieve relevant company policies, and generate grounded response recommendations.

Built as an end-to-end Data Science / NLP project using synthetic telecom customer-support data and ZENDS Communications company knowledge.

---

## 🚀 Project Overview

Customer support teams handle large volumes of queries related to billing, refunds, technical issues, complaints, and products.

This project demonstrates how AI and NLP can assist support agents by automatically:

- Understanding customer queries
- Classifying customer intent
- Detecting customer sentiment
- Determining query priority
- Retrieving relevant company information using RAG
- Generating a recommended response
- Presenting the complete analysis through an interactive Streamlit application

### Supported Customer Intents

1. Billing
2. Refund
3. Technical
4. Complaint
5. Product Inquiry

### Supported Sentiments

- 😊 Happy
- 😐 Neutral
- 😠 Angry

---

# 🏗️ System Architecture

```text
                    ┌──────────────────────────┐
                    │   ZENDS Company Docs     │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Synthetic Data Generation│
                    │       20,000 Records     │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │     NLP Intelligence     │
                    │                          │
                    │  Intent Classification   │
                    │  Sentiment Analysis      │
                    └────────────┬─────────────┘
                                 │
                                 ▼
Customer Query ───────► ┌──────────────────────┐
                        │    Query Processing   │
                        └──────────┬───────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
                 Intent        Sentiment       Priority
                    │              │              │
                    └──────────────┼──────────────┘
                                   ▼
                        ┌──────────────────────┐
                        │   RAG Knowledge Base │
                        │      ChromaDB        │
                        └──────────┬───────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │   Response Engine    │
                        │    FLAN-T5 Model     │
                        └──────────┬───────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │ Streamlit Copilot UI │
                        └──────────────────────┘
📊 Model Performance

The project includes separate evaluation pipelines for intent classification and sentiment analysis.

Component	Metric	Result
Intent Classification	Accuracy	76.3%
Sentiment Analysis	Accuracy	83.0%
RAG Retrieval	Hit@4	91.67%
Intent Classification

A fine-tuned DistilBERT model is used for five-class telecom intent classification:

Billing
Refund
Technical
Complaint
Product Inquiry

A five-class confusion matrix is included in the project evaluation artifacts.

Sentiment Analysis

A pretrained Hugging Face emotion classification model is used and mapped into three project-level sentiment categories:

joy       → Happy
neutral   → Neutral
surprise  → Neutral
anger     → Angry
disgust   → Angry
fear      → Angry
sadness   → Angry

The sentiment evaluation was performed using an independently labeled 100-message ZENDS telecom evaluation dataset.

🧠 NLP Intelligence
Intent Classification

Model: distilbert-base-uncased

The model is fine-tuned specifically for the ZENDS telecom support domain.

The classifier identifies the customer's primary support intent before the query is passed to the downstream response pipeline.

Sentiment Analysis

Model: j-hartmann/emotion-english-distilroberta-base

The pretrained emotion model is mapped into the three project categories:

Happy
Neutral
Angry

This allows the copilot to understand the customer's emotional state and provide it as context for support agents.

🔎 Retrieval-Augmented Generation (RAG)

The project uses Retrieval-Augmented Generation to ground AI responses in ZENDS Communications information.

Components
Embeddings: sentence-transformers/all-MiniLM-L6-v2
Vector Database: ChromaDB
Knowledge Source: ZENDS Communications company document
Retrieved Context: Relevant company policies and product information

The knowledge base contains information related to:

Billing
Refunds
Contracts
Service Level Agreements
Privacy
Fair Usage
Support Plans
Discounts
ZENDS Products and Services

This helps prevent the response engine from relying only on general language-model knowledge.

🤖 Response Generation

Model: google/flan-t5-small

The response engine receives:

Customer Query
      +
Intent
      +
Sentiment
      +
Priority
      +
Retrieved ZENDS Knowledge

and generates a recommended support response for the agent.

The system is designed as an agent-assistance tool, rather than an autonomous customer-service replacement.

🎯 Priority Detection

Customer priority is determined using a rule-based approach.

Priority considers factors such as:

Customer sentiment
Technical/service issues
Complaint language
Urgency indicators
Query context

The resulting priority is displayed to the support agent alongside intent and sentiment.

📚 ZENDS Communications

ZENDS Communications is a fictional global telecommunications and digital-services company created for this project.

The simulated organization provides:

📱 Mobile Connectivity
🌐 Home & Office Internet
🏢 Business Connectivity
☁️ Cloud & Data Center Services
📡 IoT & Smart Solutions

The project uses the company's simulated policies and knowledge base to demonstrate a realistic enterprise AI support workflow.

🖥️ Streamlit Application

The project includes an interactive Streamlit interface with three main sections:

💬 Copilot

The primary customer-support workspace.

Agents can enter customer messages and receive:

Intent
Sentiment
Priority
Recommended response
Retrieved knowledge
Supporting source details
📊 Model Analytics

Provides visibility into:

Intent model performance
Sentiment model performance
Intent confusion matrix
NLP processing pipeline
ℹ️ About Project

Provides an overview of:

Project objective
AI pipeline
Technologies used
End-to-end workflow
🛠️ Technology Stack
Programming
Python
Data Science
Pandas
NumPy
Scikit-learn
Matplotlib
NLP / Machine Learning
Hugging Face Transformers
DistilBERT
DistilRoBERTa
FLAN-T5
Sentence Transformers
RAG
ChromaDB
Database
MySQL
Application
Streamlit
Development
Jupyter Notebook
VS Code
Git
GitHub
📁 Project Structure
Copilot Project/
│
├── 1_Data_Foundation/
│   ├── code/
│   ├── config/
│   ├── data/
│   └── evaluation/
│
├── 2_NLP_Intelligence/
│   ├── code/
│   ├── data/
│   ├── models/
│   ├── scripts/
│   └── evaluation/
│
├── 3_RAG_Knowledge/
│   ├── code/
│   ├── data/
│   ├── vector_db/
│   └── evaluation/
│
├── 4_AI_Response_Engine/
│   ├── code/
│   └── evaluation/
│
├── 5_Streamlit_Integration/
│   ├── app.py
│   ├── ui_helpers.py
│   ├── tests/
│   └── README.md
│
├── scripts/
├── tests/
├── requirements.txt
├── README.md
└── .gitignore
🔄 End-to-End Workflow
Customer Message
       ↓
Text Preprocessing
       ↓
Intent Classification
       ↓
Sentiment Analysis
       ↓
Priority Detection
       ↓
Knowledge Retrieval
       ↓
Context + Query
       ↓
Response Generation
       ↓
Recommended Agent Response
       ↓
Streamlit Interface
🧪 Evaluation

The project includes evaluation artifacts for the major AI components.

Intent Evaluation

The intent classifier is evaluated using:

Accuracy
Precision
Recall
F1 Score
Five-class confusion matrix
Sentiment Evaluation

The sentiment model was evaluated using 100 manually labeled telecom support messages.

Dataset distribution:

Happy    : 33
Neutral  : 34
Angry    : 33
Total    : 100

Result:

Sentiment Accuracy: 83.0%

RAG Evaluation

The retrieval system was evaluated using targeted telecom knowledge queries.

Hit@4: 91.67%

▶️ Running the Project Locally
1. Clone the repository
git clone https://github.com/ANBALAGHANRATHINAM/AI-Customer-Support-Copilot.git
cd Ai-Customer-Support-Copilot
2. Create a virtual environment
python -m venv .venv
Windows
.venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
4. Run the Streamlit application
streamlit run 5_Streamlit_Integration/app.py

The application will open in your browser.

🔐 Data & Model Notes

This repository intentionally excludes large local model files, checkpoints, virtual environments, and runtime vector database files through .gitignore.

The project is structured so that development artifacts and large generated files are kept outside the Git repository.

📌 Key Project Highlights
End-to-end AI customer support pipeline
20,000-record synthetic telecom dataset
Five-class intent classification
Pretrained sentiment analysis
Rule-based priority detection
Retrieval-Augmented Generation
ChromaDB vector search
Grounded response generation
Streamlit AI copilot interface
Model evaluation and confusion matrix
Modular multi-stage architecture
Production-style project structure
👨‍💻 Author

Madan Kumar S

Data Science | Machine Learning | NLP | SQL | Python
