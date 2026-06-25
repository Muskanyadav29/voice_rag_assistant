import os
import tempfile
import base64
import winsound
from pathlib import Path
import speech_recognition as sr
import pyttsx3
import google.generativeai as genai
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# ==========================
# Load API Keys and Config
# ==========================
# Force override of any environment variables using the .env file in the backend
SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(SCRIPT_DIR / "backend" / ".env", override=True)
load_dotenv(SCRIPT_DIR / ".env", override=True)

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "GOOGLE_API_KEY")

# Initialize Gemini if configured (for fallback)
gemini_model = None
if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY":
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    except Exception as e:
        print(f"Warning: Failed to initialize Gemini fallback: {e}")

# Initialize Sarvam Client
sarvam_client = None
if SARVAM_API_KEY:
    try:
        from sarvamai import SarvamAI
        sarvam_client = SarvamAI(api_subscription_key=SARVAM_API_KEY)
        masked = SARVAM_API_KEY[:6] + "..." + SARVAM_API_KEY[-4:] if len(SARVAM_API_KEY) > 10 else "..."
        print(f"Loaded Sarvam API Key: {masked}")
    except Exception as e:
        print(f"Warning: Failed to initialize Sarvam client: {e}")
else:
    print("Warning: SARVAM_API_KEY is not configured in backend/.env")

# ==========================
# Text To Speech (TTS)
# ==========================
pyttsx_engine = pyttsx3.init()

def speak(text):
    print("\nAssistant:", text)
    
    # Try Sarvam TTS first
    if sarvam_client:
        try:
            response = sarvam_client.text_to_speech.convert(
                target_language_code="en-IN",
                text=text[:2500],
                model="bulbul:v3",
                speaker="shubh",
                output_audio_codec="wav",
            )
            if response.audios:
                audio_data = base64.b64decode(response.audios[0])
                winsound.PlaySound(audio_data, winsound.SND_MEMORY)
                return
        except Exception as e:
            print(f"(Sarvam TTS failed, falling back to offline TTS: {e})")
            
    # Fallback to local pyttsx3 engine
    pyttsx_engine.say(text)
    pyttsx_engine.runAndWait()

# ==========================
# Speech To Text (STT)
# ==========================
def listen():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("\nSpeak your question...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    # Try Sarvam STT first
    if sarvam_client:
        try:
            # Write audio to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio.get_wav_data())
                temp_filename = f.name
            
            try:
                with open(temp_filename, "rb") as audio_file:
                    response = sarvam_client.speech_to_text.transcribe(
                        file=audio_file,
                        model="saaras:v3",
                        language_code="en-IN",
                        input_audio_codec="wav",
                    )
                text = response.transcript.strip()
                if text:
                    print("You (Sarvam STT):", text)
                    return text
            finally:
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
        except Exception as e:
            print(f"(Sarvam STT failed: {e}. Falling back to Google STT...)")

    # Fallback to Google speech recognition
    try:
        text = recognizer.recognize_google(audio)
        print("You (Google STT):", text)
        return text
    except Exception:
        return ""

# ==========================
# Sample Knowledge Base
# ==========================
docs = [
    Document(
        page_content="""
        Sarvam AI provides speech recognition,
        text to speech and language AI services.
        """
    ),

    Document(
        page_content="""
        RAG stands for Retrieval Augmented Generation.
        It retrieves relevant context before generating responses.
        """
    ),

    Document(
        page_content="""
        Python is a powerful programming language
        used in AI and machine learning.
        """
    )
]

# ==========================
# Embedding Model & Vector DB
# ==========================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb = Chroma.from_documents(
    docs,
    embeddings
)

# ==========================
# Retrieve Context
# ==========================
def retrieve(query):
    results = vectordb.similarity_search(
        query,
        k=2
    )
    context = "\n".join(
        [doc.page_content for doc in results]
    )
    return context

# ==========================
# Generate Answer (LLM)
# ==========================
def answer_question(question):
    context = retrieve(question)

    # Try Sarvam LLM first
    if sarvam_client:
        try:
            system_prompt = (
                "You are a helpful Voice RAG assistant. Answer clearly and concisely. "
                "Use the provided document context when it is relevant. "
                "If the context does not contain the answer, use your general knowledge."
            )
            user_prompt = f"Context from uploaded documents:\n{context or 'No document context available.'}\n\nQuestion:\n{question}"
            
            response = sarvam_client.chat.completions(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model="sarvam-m",
                temperature=0.3,
                max_tokens=1024,
            )
            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"(Sarvam Chat Completion failed, falling back to Gemini: {e})")

    # Fallback to Google Gemini
    if gemini_model:
        try:
            prompt = f"""
            Use the context below to answer.

            Context:
            {context}

            Question:
            {question}

            If answer is not present in context,
            answer using your general knowledge.
            """
            response = gemini_model.generate_content(prompt)
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            print(f"(Gemini fallback failed: {e}. Falling back to local RAG extractor...)")

    # Fallback to Local RAG Extractor (so the script never crashes on key issues)
    if context and context.strip():
        return (
            f"Here is the matching context found in your documents:\n\n"
            f"\"{context.strip()}\"\n\n"
            f"[⚠️ Note: Both Sarvam and Gemini API keys are missing or invalid. Add a valid API key to backend/.env for full AI answers.]"
        )
    else:
        return (
            f"I couldn't find any relevant document context for: \"{question}\".\n\n"
            f"[⚠️ Note: Both Sarvam and Gemini API keys are missing or invalid. Add a valid API key to backend/.env for full AI answers.]"
        )

# ==========================
# Main Program
# ==========================
def main():
    speak("Hello, I am your Voice RAG Assistant.")

    while True:
        query = listen()

        if query == "":
            continue

        if query.lower() in ["exit", "quit", "stop"]:
            speak("Goodbye")
            break

        answer = answer_question(query)
        speak(answer)

if __name__ == "__main__":
    main()