# 🎵 Song Lyrics Emotion Detection AI

A data-driven full-stack web application that predicts emotions from song lyrics using a BERT-based model. It visualizes the emotional fingerprint of lyrics using Russell’s Circumplex Model (Valence vs. Arousal).

## Features
- **Emotion Analysis**: Multi-class classification (Happy, Sad, Angry, Fear, Surprise, Love/Calm).
- **Visualization**: Interactive Scatter Plot of Valence and Arousal.
- **Modern UI**: Glassmorphism design with React and Tailwind-like styling.
- **Backend API**: Flask REST API using HuggingFace Transformers.

## Project Structure
\`\`\`
emotion_lyrics_app/
├── backend/            # Flask API
│   ├── app.py          # Application entry point
│   └── requirements.txt
├── frontend/           # React Frontend (Vite)
│   ├── src/
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
└── model/              # Model Training Scripts
    ├── train.py        # Fine-tune BERT
    └── evaluate.py     # Generate metrics & Confusion Matrix
\`\`\`

## 🚀 How to Run Locally

### Prerequisites
- Python 3.8+
- Node.js 16+

### 1. Backend Setup
1. Open a terminal and navigate to the `backend` folder:
   \`\`\`bash
   cd backend
   \`\`\`
2. Install Python dependencies:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`
3. Start the Flask server:
   \`\`\`bash
   python app.py
   \`\`\`
   The server will start at `http://127.0.0.1:5000`.

### 2. Frontend Setup
1. Open a **new** terminal and navigate to the `frontend` folder:
   \`\`\`bash
   cd frontend
   \`\`\`
2. Install Node dependencies:
   \`\`\`bash
   npm install
   \`\`\`
3. Start the development server:
   \`\`\`bash
   npm run dev
   \`\`\`
4. Open the link shown (usually `http://localhost:5173`) in your browser.

## 🧠 Model Training (Optional)
The backend uses a pre-trained model by default. If you want to fine-tune your own BERT model:

1. Navigate to the `model` folder.
2. Run the training script (requires GPU recommended):
   \`\`\`bash
   python train.py
   \`\`\`
3. This will save the fine-tuned model to `./emotion_model`.
4. Run evaluation:
   \`\`\`bash
   python evaluate.py
   \`\`\`

## ☁️ Azure Deployment (Optional)

### Backend (Azure App Service)
1. **Create an App Service** (Python 3.10) in Azure Portal.
2. **Deploy Code**: Use VS Code Azure Tools extension or standard Git deployment.
   - Run `az webapp up --sku F1 --name <your-app-name>` inside the `backend` folder.
3. **Startup Command**: Set the startup command in Configuration -> General Settings:
   \`\`\`bash
   gunicorn --bind=0.0.0.0:8000 app:app
   \`\`\`

### Frontend (Azure Static Web Apps)
1. **Create Static Web App** in Azure Portal.
2. **Link GitHub**: Connect your repository.
3. **Build Presets**:
   - **App location**: `frontend`
   - **Output location**: `dist`
4. **Environment Variables**: Update your `fetch` URL in `App.jsx` to point to your deployed backend URL instead of localhost.

---
Built with ❤️ using React, Flask, and Transformers.
