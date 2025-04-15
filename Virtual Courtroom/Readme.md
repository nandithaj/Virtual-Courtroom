# 🏛️ Virtual Courtroom

The **Virtual Courtroom** is an AI-powered platform designed to simulate real-world criminal court proceedings, aiming to streamline and automate key legal processes using modern Machine Learning (ML) and Natural Language Processing (NLP) technologies. The system integrates document analysis, audio transcription, slot booking, and verdict prediction into a seamless judicial experience.

## 🚀 Key Features

- **Slot Booking System**: Prosecutors can book hearing slots and notify the defense with secure credentials. The defense then selects an available time slot, confirmed via email.
- **Petition Handling**: Petitions are uploaded by the prosecutor, processed using PaddleOCR for text extraction, and made available to all stakeholders securely.
- **Cross Examination Module**: Real-time courtroom dialogue is transcribed using Whisper and PyAnnote, capturing speaker segments accurately.
- **Legal Judgment Prediction**: The transcript is summarized using a T5 model. A binary classifier checks the innocence or guilt of the accused. If guilty, the system predicts applicable IPC sections using SBERT + FAISS and generates a structured verdict.

## ⚙️ Tech Stack

- **Frontend**: Flutter  
- **Backend**: Flask  
- **Database**: PostgreSQL  
- **OCR**: PaddleOCR  
- **Speech Recognition**: Whisper, PyAnnote  
- **NLP Models**: T5 for summarization, custom Binary Classifier, SBERT + FAISS for legal reasoning  
- **Email Integration**: SMTP  
- **File Storage**: Google Drive API  

## 📌 Objective

To reduce case backlogs, procedural delays, and human error by automating routine tasks in legal proceedings while supporting judges and legal professionals with AI-assisted insights.

## 🧠 ML/NLP Highlights

- **Speech-to-Text**: Converts courtroom discussions into structured text for further analysis.
- **Summarization**: T5 model extracts key elements from lengthy transcripts.
- **Verdict Generation**: Combines classification and semantic search (via SBERT) to predict relevant IPC sections and punishments.

## 🔐 Assumptions

- Inputs (e.g., petitions) are in standard formats like PDF/DOCX.
- Audio input is clear and conducted in a quiet environment.
- The ML models rely on structured and relevant case information.
- Current version handles only criminal cases and uses English language.

## 📈 Future Scope

- Expand to support civil and constitutional cases.
- Introduce multi-defendant support and multi-language processing.
- Improve input authentication and document verification mechanisms.
