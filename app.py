# import streamlit
# import openai
# import json
# import time
# import pandas as pd
# import plotly.graph_objects as go
# import plotly.express as px
# from datetime import datetime, timedelta
# import sqlite3
# import hashlib
# from textblob import TextBlob
# import random
# import re
# from typing import Dict, List, Tuple
# import requests

# # Configure Streamlit page
# st.set_page_config(
#     page_title="AI Mentor - Personalized Learning Coach",
#     page_icon="🤖",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for better UI
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 5rem;
#         color: #2E86AB;
#         text-align: center;
#         margin-bottom: 3rem;
#         text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
#         padding: 1rem;
#         background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
#         border-radius: 10px;
#         box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                
            
#     }
#     .chat-message {
#         padding: 1rem;
#         border-radius: 10px;
#         margin: 1rem 0;
#         box-shadow: 0 2px 4px rgba(0,0,0,0.1);
#     }
#     .user-message {
#         background-color: #E3F2FD;
#         border-left: 4px solid #2196F3;
#     }
#     .ai-message {
#         background-color: #F3E5F5;
#         border-left: 4px solid #9C27B0;
#     }
#     .emotion-indicator {
#         padding: 0.5rem;
#         border-radius: 20px;
#         text-align: center;
#         margin: 0.5rem 0;
#         font-weight: bold;
#     }
#     .confident { background-color: #C8E6C9; color: #2E7D32; }
#     .confused { background-color: #FFECB3; color: #F57C00; }
#     .frustrated { background-color: #FFCDD2; color: #C62828; }
#     .excited { background-color: #E1BEE7; color: #7B1FA2; }
#     .progress-card {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         padding: 1rem;
#         border-radius: 10px;
#         margin: 1rem 0;
#     }
#     .flashcard {
#         background-color: #FFF3E0;
#         border: 2px solid #FF9800;
#         border-radius: 10px;
#         padding: 1rem;
#         margin: 1rem 0;
#         cursor: pointer;
#         transition: transform 0.2s;
#     }
#     .flashcard:hover {
#         transform: scale(1.02);
#     }
# </style>
# """, unsafe_allow_html=True)

# # Database setup
# def init_database():
#     conn = sqlite3.connect('ai_mentor.db')
#     cursor = conn.cursor()
    
#     # Users table
#     cursor.execute('''
#         CREATE TABLE IF NOT EXISTS users (
#             id INTEGER PRIMARY KEY,
#             username TEXT UNIQUE,
#             password_hash TEXT
#             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#         )
#     ''')
    
#     # Learning sessions table
#     cursor.execute('''
#         CREATE TABLE IF NOT EXISTS sessions (
#             id INTEGER PRIMARY KEY,
#             user_id INTEGER,
#             topic TEXT,
#             start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#             end_time TIMESTAMP,
#             total_questions INTEGER DEFAULT 0,
#             correct_answers INTEGER DEFAULT 0,
#             avg_sentiment REAL DEFAULT 0.0,
#             difficulty_level INTEGER DEFAULT 3,
#             FOREIGN KEY (user_id) REFERENCES users (id)
#         )
#     ''')
    
#     # Conversations table
#     cursor.execute('''
#         CREATE TABLE IF NOT EXISTS conversations (
#             id INTEGER PRIMARY KEY,
#             session_id INTEGER,
#             user_message TEXT,
#             ai_response TEXT,
#             sentiment_score REAL,
#             emotion_detected TEXT,
#             response_time REAL,
#             is_correct BOOLEAN,
#             difficulty_level INTEGER,
#             timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#             FOREIGN KEY (session_id) REFERENCES sessions (id)
#         )
#     ''')
    
#     # Flashcards table
#     cursor.execute('''
#         CREATE TABLE IF NOT EXISTS flashcards (
#             id INTEGER PRIMARY KEY,
#             user_id INTEGER,
#             topic TEXT,
#             question TEXT,
#             answer TEXT,
#             difficulty INTEGER,
#             times_reviewed INTEGER DEFAULT 0,
#             last_reviewed TIMESTAMP,
#             mastery_level REAL DEFAULT 0.0,
#             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#             FOREIGN KEY (user_id) REFERENCES users (id)
#         )
#     ''')
    
#     conn.commit()
#     conn.close()

# # Sentiment Analysis Engine
# class SentimentAnalyzer:
#     def __init__(self):
#         self.emotion_keywords = {
#             'confused': ['confused', 'lost', 'don\'t understand', 'unclear', 'what', 'huh', '?'],
#             'frustrated': ['frustrated', 'annoying', 'difficult', 'hard', 'stuck', 'can\'t'],
#             'confident': ['yes', 'got it', 'understand', 'clear', 'easy', 'sure'],
#             'excited': ['cool', 'awesome', 'interesting', 'wow', 'amazing', 'love']
#         }
    
#     def analyze_emotion(self, text: str, response_time: float = None, is_correct: bool = None) -> Dict:
#         # Text-based sentiment analysis
#         blob = TextBlob(text.lower())
#         polarity = blob.sentiment.polarity
#         subjectivity = blob.sentiment.subjectivity
        
#         # Keyword-based emotion detection
#         emotion_scores = {}
#         for emotion, keywords in self.emotion_keywords.items():
#             score = sum(1 for keyword in keywords if keyword in text.lower())
#             emotion_scores[emotion] = score
        
#         # Determine primary emotion
#         primary_emotion = max(emotion_scores, key=emotion_scores.get) if max(emotion_scores.values()) > 0 else 'neutral'
        
#         # Adjust based on correctness and response time
#         if is_correct is not None:
#             if is_correct and primary_emotion == 'neutral':
#                 primary_emotion = 'confident'
#             elif not is_correct and primary_emotion == 'neutral':
#                 primary_emotion = 'confused'
        
#         if response_time and response_time > 30:  # Long response time might indicate confusion
#             if primary_emotion in ['neutral', 'confident']:
#                 primary_emotion = 'confused'
        
#         return {
#             'emotion': primary_emotion,
#             'polarity': polarity,
#             'subjectivity': subjectivity,
#             'confidence': max(emotion_scores.values()) / len(text.split()) if text else 0
#         }

# # Adaptive Difficulty Engine
# class DifficultyAdapter:
#     def __init__(self):
#         self.min_difficulty = 1
#         self.max_difficulty = 10
#         self.adjustment_factors = {
#             'confident': 0.5,
#             'excited': 0.3,
#             'confused': -0.7,
#             'frustrated': -1.0,
#             'neutral': 0.0
#         }
    
#     def adjust_difficulty(self, current_difficulty: int, emotion: str, accuracy: float, response_time: float) -> int:
#         adjustment = 0
        
#         # Emotion-based adjustment
#         adjustment += self.adjustment_factors.get(emotion, 0)
        
#         # Performance-based adjustment
#         if accuracy > 0.8:
#             adjustment += 0.3
#         elif accuracy < 0.5:
#             adjustment -= 0.5
        
#         # Response time adjustment
#         if response_time < 10:  # Quick response
#             adjustment += 0.2
#         elif response_time > 45:  # Slow response
#             adjustment -= 0.3
        
#         new_difficulty = current_difficulty + adjustment
#         return max(self.min_difficulty, min(self.max_difficulty, round(new_difficulty)))

# # AI Tutor Engine
# # class AITutor:
# #     def __init__(self, api_key: str):
# #         openai.api_key = api_key
# #         self.conversation_history = []
# #         self.current_topic = None
# #         self.difficulty_level = 3
        
# #     def set_topic(self, topic: str):
# #         self.current_topic = topic
# #         self.conversation_history = []
    
# #     def generate_system_prompt(self, emotion: str, difficulty: int) -> str:
# #         base_prompt = f"""You are an encouraging, patient AI tutor specializing in {self.current_topic}. 
# #         Current difficulty level: {difficulty}/10.
# #         Student's current emotional state: {emotion}.
        
# #         Guidelines:
# #         - Adapt your teaching style based on the student's emotional state
# #         - If confused/frustrated: Use simpler language, more examples, break down concepts
# #         - If confident/excited: Increase complexity, introduce advanced concepts
# #         - Always be encouraging and supportive
# #         - Use analogies and real-world examples
# #         - Ask follow-up questions to check understanding
# #         - Generate practice questions appropriate to the difficulty level
# #         """
        
# #         emotion_adaptations = {
# #             'confused': "The student seems confused. Use very simple language, provide step-by-step explanations, and give concrete examples.",
# #             'frustrated': "The student is frustrated. Be extra encouraging, suggest taking a break if needed, and simplify the concept significantly.",
# #             'confident': "The student is confident. You can introduce more challenging concepts and ask deeper questions.",
# #             'excited': "The student is excited! Feed their enthusiasm with interesting facts and advanced applications.",
# #             'neutral': "The student seems neutral. Engage them with interesting examples and check their understanding."
# #         }
        
# #         return base_prompt + "\n" + emotion_adaptations.get(emotion, "")
    
# #     def generate_response(self, user_input: str, emotion: str, context: str = "") -> str:
# #         try:
# #             system_prompt = self.generate_system_prompt(emotion, self.difficulty_level)
            
# #             messages = [
# #                 {"role": "system", "content": system_prompt},
# #                 {"role": "user", "content": f"Context: {context}\n\nStudent says: {user_input}"}
# #             ]
            
# #             response = openai.ChatCompletion.create(
# #                 model="gpt-3.5-turbo",  # Use gpt-4 if available
# #                 messages=messages,
# #                 max_tokens=300,
# #                 temperature=0.7
# #             )
            
# #             return response.choices[0].message.content
# #         except Exception as e:
# #             return f"I'm having trouble connecting right now. Let me try to help anyway: {str(e)}"
    
# #     def generate_quiz_question(self, topic: str, difficulty: int) -> Dict:
# #         try:
# #             prompt = f"""Generate a {difficulty}/10 difficulty quiz question about {topic}.
# #             Return ONLY a JSON object with this exact format:
# #             {{
# #                 "question": "Your question here",
# #                 "options": ["A) option1", "B) option2", "C) option3", "D) option4"],
# #                 "correct_answer": "A",
# #                 "explanation": "Brief explanation of why this is correct"
# #             }}"""
            
# #             response = openai.ChatCompletion.create(
# #                 model="gpt-3.5-turbo",
# #                 messages=[{"role": "user", "content": prompt}],
# #                 max_tokens=200,
# #                 temperature=0.7
# #             )
            
# #             # Parse JSON response
# #             content = response.choices[0].message.content.strip()
# #             if content.startswith('```json'):
# #                 content = content[7:-3]
# #             elif content.startswith('```'):
# #                 content = content[3:-3]
            
# #             return json.loads(content)
# #         except Exception as e:
# #             # Fallback question
# #             return {
# #                 "question": f"What is an important concept in {topic}?",
# #                 "options": ["A) Concept 1", "B) Concept 2", "C) Concept 3", "D) Concept 4"],
# #                 "correct_answer": "A",
# #                 "explanation": "This is a basic concept in the subject."
# #             }
# from transformers import pipeline, set_seed

# generator = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")
# set_seed(42)

# class AITutor:
#     def __init__(self):
#         self.conversation_history = []
#         self.current_topic = None
#         self.difficulty_level = 3
        
#     def set_topic(self, topic: str):
#         self.current_topic = topic
#         self.conversation_history = []

#     # def generate_response(self, user_input: str, emotion: str, context: str = "") -> str:
#     #     prompt = f"You are a helpful tutor teaching {self.current_topic}. The student seems {emotion}. They asked: {user_input}\nTutor:"
#     #     try:
#     #         result = generator(prompt, max_length=200, num_return_sequences=1)
#     #         return result[0]["generated_text"]
#     #     except Exception as e:
#     #         return f"Sorry, I ran into an error: {str(e)}"
#     def generate_response(self, user_input: str, emotion: str, context: str = "") -> str:
#         prompt = f"Student Question: {user_input}\nTutor Answer:"
#         try:
#            output = generator(prompt, max_length=150, num_return_sequences=1)
#            generated = output[0]["generated_text"]
#         # Extract only the part after "Tutor Answer:"
#            if "Tutor Answer:" in generated:
#                answer = generated.split("Tutor Answer:")[-1].strip()
#            else:
#                answer = generated.strip()
#            return answer
#         except Exception as e:
#           return f"Sorry, I ran into an error: {str(e)}"


#     def generate_quiz_question(self, topic: str, difficulty: int) -> Dict:
#         # Static fallback quiz for demo purposes
#         return {
#             "question": f"What is a key concept in {topic}?",
#             "options": ["A) Concept A", "B) Concept B", "C) Concept C", "D) Concept D"],
#             "correct_answer": "A",
#             "explanation": "This is one of the basic concepts."
#         }


# # Flashcard Generator
# class FlashcardGenerator:
#     @staticmethod
#     def generate_flashcards(topic: str, conversation_content: str, difficulty: int) -> List[Dict]:
#         # Simple keyword extraction for flashcard generation
#         keywords = re.findall(r'\b[A-Z][a-z]{3,}\b', conversation_content)
#         unique_keywords = list(set(keywords))[:5]  # Limit to 5 flashcards
        
#         flashcards = []
#         for keyword in unique_keywords:
#             flashcard = {
#                 'question': f"What is {keyword} in the context of {topic}?",
#                 'answer': f"Key concept related to {topic}",
#                 'difficulty': difficulty,
#                 'topic': topic
#             }
#             flashcards.append(flashcard)
        
#         return flashcards

# # Database Helper Functions
# def get_user_id(username: str) -> int:
#     conn = sqlite3.connect('ai_mentor.db')
#     cursor = conn.cursor()
    
#     cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
#     result = cursor.fetchone()
    
#     if result:
#         user_id = result[0]
#     else:
#         cursor.execute('INSERT INTO users (username) VALUES (?)', (username,))
#         user_id = cursor.lastrowid
#         conn.commit()
    
#     conn.close()
#     return user_id

# def create_session(user_id: int, topic: str) -> int:
#     conn = sqlite3.connect('ai_mentor.db')
#     cursor = conn.cursor()
    
#     cursor.execute('INSERT INTO sessions (user_id, topic) VALUES (?, ?)', (user_id, topic))
#     session_id = cursor.lastrowid
#     conn.commit()
#     conn.close()
    
#     return session_id

# def hash_password(password: str) -> str:
#     return hashlib.sha256(password.encode()).hexdigest()

# def verify_user(username: str, password: str) -> bool:
#     conn = sqlite3.connect('ai_mentor.db')
#     cursor = conn.cursor()
#     cursor.execute('SELECT password_hash FROM users WHERE username = ?', (username,))
#     result = cursor.fetchone()
#     conn.close()
#     if result:
#         return result[0] == hash_password(password)
#     return False

# def register_user(username: str, password: str) -> bool:
#     conn = sqlite3.connect('ai_mentor.db')
#     cursor = conn.cursor()
#     try:
#         cursor.execute(
#             'INSERT INTO users (username, password_hash) VALUES (?, ?)',
#             (username, hash_password(password))
#         )
#         conn.commit()
#         return True
#     except sqlite3.IntegrityError:
#         return False
#     finally:
#         conn.close()


# def save_conversation(session_id: int, user_message: str, ai_response: str, 
#                      sentiment_score: float, emotion: str, response_time: float,
#                      is_correct: bool = None, difficulty: int = 3):
#     conn = sqlite3.connect('ai_mentor.db')
#     cursor = conn.cursor()
    
#     cursor.execute('''
#         INSERT INTO conversations 
#         (session_id, user_message, ai_response, sentiment_score, emotion_detected, 
#          response_time, is_correct, difficulty_level)
#         VALUES (?, ?, ?, ?, ?, ?, ?, ?)
#     ''', (session_id, user_message, ai_response, sentiment_score, emotion, 
#           response_time, is_correct, difficulty))
    
#     conn.commit()
#     conn.close()

# def get_user_analytics(user_id: int) -> Dict:
#     conn = sqlite3.connect('ai_mentor.db')
#     cursor = conn.cursor()
    
#     # Get session stats
#     cursor.execute('''
#         SELECT COUNT(*) as total_sessions,
#                AVG(correct_answers * 1.0 / total_questions) as avg_accuracy,
#                AVG(avg_sentiment) as avg_sentiment
#         FROM sessions WHERE user_id = ?
#     ''', (user_id,))
    
#     session_stats = cursor.fetchone()
    
#     # Get emotion distribution
#     cursor.execute('''
#         SELECT emotion_detected, COUNT(*) as count
#         FROM conversations c
#         JOIN sessions s ON c.session_id = s.id
#         WHERE s.user_id = ?
#         GROUP BY emotion_detected
#     ''', (user_id,))
    
#     emotion_data = cursor.fetchall()
    
#     conn.close()
    
#     return {
#         'total_sessions': session_stats[0] or 0,
#         'avg_accuracy': session_stats[1] or 0,
#         'avg_sentiment': session_stats[2] or 0,
#         'emotion_distribution': dict(emotion_data) if emotion_data else {}
#     }

# # Initialize components
# @st.cache_resource
# def initialize_components():
#     init_database()
#     sentiment_analyzer = SentimentAnalyzer()
#     difficulty_adapter = DifficultyAdapter()
#     return sentiment_analyzer, difficulty_adapter

# # Main Application
# def main():
#     def login_ui():
#        st.subheader("🔐 Login to AI Mentor")
#        username = st.text_input("Username")
#        password = st.text_input("Password", type="password")
#        if st.button("Login"):
#            if verify_user(username, password):
#                st.session_state.logged_in = True
#                st.session_state.username = username
#                st.success("✅ Logged in successfully!")
#                st.rerun()
#            else:
#                st.error("❌ Invalid credentials.")

#     def register_ui():
#        st.subheader("🆕 Register New User")
#        username = st.text_input("New Username")
#        password = st.text_input("New Password", type="password")
#        if st.button("Register"):
#            if register_user(username, password):
#                st.success("✅ Registered successfully! Please login.")
#            else:
#               st.error("❌ Username already exists.")

#     def auth_page():
#         st.markdown("## 👋 Welcome to AI Mentor Login Portal")
#         auth_tab = st.radio("Select Option", ["Login", "Register"])
#         if auth_tab == "Login":
#             login_ui()
#         else:
#            register_ui()

#     st.markdown('<h1 class="main-header">🤖 AI Mentor</h1>', unsafe_allow_html=True)
#     st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Your Personalized Learning Coach with Emotion & Performance Awareness</p>', unsafe_allow_html=True)
    
#     # Initialize components
#     sentiment_analyzer, difficulty_adapter = initialize_components()
    
#     # Sidebar for navigation and settings
#     with st.sidebar:
#         st.header("🎯 Learning Dashboard")
        
#         # User authentication (simplified)
#         if 'username' not in st.session_state:
#             st.session_state.username = st.text_input("Enter your username:", value="student_demo")
        
#         if st.session_state.username:
#             user_id = get_user_id(st.session_state.username)
#             st.success(f"Welcome, {st.session_state.username}! 👋")
            
#             # Topic selection
#             topics = [
#                 "Newton's Laws of Physics",
#                 "Algebra Fundamentals", 
#                 "Basic Chemistry",
#                 "Biology Cells",
#                 "Calculus Derivatives",
#                 "Statistics & Probability",
#                 "Computer Science Algorithms",
#                 "Geometry Theorems"
#             ]
            
#             selected_topic = st.selectbox("Choose a topic to learn:", topics)
            
#             # OpenAI API Key input
#             # api_key = st.text_input("OpenAI API Key:", type="password", 
#             #                        help="Enter your OpenAI API key to enable AI features")
#             api_key="AIzaSyDC22K5GOgiVOzIy04QLTNEEg8dcxtldsk"
#             if api_key and selected_topic:
#                 # Initialize AI tutor
#                 if 'ai_tutor' not in st.session_state or st.session_state.get('current_topic') != selected_topic:
#                     st.session_state.ai_tutor = AITutor()
#                     st.session_state.ai_tutor.set_topic(selected_topic)
#                     st.session_state.current_topic = selected_topic
#                     st.session_state.session_id = create_session(user_id, selected_topic)
#                     st.session_state.difficulty_level = 3
#                     st.session_state.questions_asked = 0
#                     st.session_state.correct_answers = 0
                
#                 # Display current stats
#                 st.markdown("### 📊 Current Session")
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     st.metric("Difficulty", f"{st.session_state.get('difficulty_level', 3)}/10")
#                 with col2:
#                     accuracy = (st.session_state.get('correct_answers', 0) / max(st.session_state.get('questions_asked', 1), 1)) * 100
#                     st.metric("Accuracy", f"{accuracy:.1f}%")
                
#                 # User analytics
#                 analytics = get_user_analytics(user_id)
#                 st.markdown("### 📈 Your Progress")
#                 st.metric("Total Sessions", analytics['total_sessions'])
#                 st.metric("Overall Accuracy", f"{analytics['avg_accuracy']*100:.1f}%")
                
#                 # Emotion distribution chart
#                 if analytics['emotion_distribution']:
#                     fig = px.pie(
#                         values=list(analytics['emotion_distribution'].values()),
#                         names=list(analytics['emotion_distribution'].keys()),
#                         title="Your Learning Emotions"
#                     )
#                     fig.update_layout(height=300)
#                     st.plotly_chart(fig, use_container_width=True)
    
#     # Main content area
#     if st.session_state.get('username') and st.session_state.get('ai_tutor'):
#         # Create tabs for different features
#         tab1, tab2, tab3, tab4 = st.tabs(["💬 AI Tutor Chat", "📝 Practice Quiz", "🎴 Flashcards", "📊 Analytics"])
        
#         with tab1:
#             st.markdown("### 🤖 Chat with Your AI Mentor")
            
#             # Initialize chat history
#             if 'chat_history' not in st.session_state:
#                 st.session_state.chat_history = []
#                 # Welcome message
#                 welcome_msg = f"Hello! I'm your AI mentor for {st.session_state.current_topic}. I'm here to help you learn at your own pace. What would you like to know?"
#                 st.session_state.chat_history.append({"role": "ai", "content": welcome_msg, "emotion": "neutral"})
            
#             # Display chat history
#             for message in st.session_state.chat_history:
#                 if message["role"] == "user":
#                     with st.container():
#                         st.markdown(f'<div class="chat-message user-message">{message["content"]}</div>', unsafe_allow_html=True)
#                         if "emotion" in message:
#                             emotion_class = message["emotion"]
#                             st.markdown(f'<div class="emotion-indicator {emotion_class}">Detected: {emotion_class.title()}</div>', unsafe_allow_html=True)
#                 else:
#                     with st.container():
#                         st.markdown(f'<div class="chat-message ai-message">{message["content"]}</div>', unsafe_allow_html=True)
            
#             # Chat input
#             user_input = st.text_input("Ask me anything about the topic:", key="chat_input")
            
#             if user_input and st.button("Send", key="send_chat"):
#                 start_time = time.time()
                
#                 # Analyze sentiment
#                 emotion_analysis = sentiment_analyzer.analyze_emotion(user_input)
#                 emotion = emotion_analysis['emotion']
#                 sentiment_score = emotion_analysis['polarity']
                
#                 response_time = time.time() - start_time
                
#                 # Generate AI response
#                 ai_response = st.session_state.ai_tutor.generate_response(
#                     user_input, emotion, context=st.session_state.current_topic
#                 )
                
#                 # Add to chat history
#                 st.session_state.chat_history.append({
#                     "role": "user", 
#                     "content": user_input, 
#                     "emotion": emotion
#                 })
#                 st.session_state.chat_history.append({
#                     "role": "ai", 
#                     "content": ai_response
#                 })
                
#                 # Save to database
#                 save_conversation(
#                     st.session_state.session_id,
#                     user_input,
#                     ai_response,
#                     sentiment_score,
#                     emotion,
#                     response_time
#                 )
                
#                 # Adjust difficulty based on emotion
#                 current_accuracy = st.session_state.get('correct_answers', 0) / max(st.session_state.get('questions_asked', 1), 1)
#                 new_difficulty = difficulty_adapter.adjust_difficulty(
#                     st.session_state.get('difficulty_level', 3),
#                     emotion,
#                     current_accuracy,
#                     response_time
#                 )
#                 st.session_state.difficulty_level = new_difficulty
#                 st.session_state.ai_tutor.difficulty_level = new_difficulty
                
#                 st.rerun()
        
#         with tab2:
#             st.markdown("### 📝 Practice Quiz")
#             st.markdown("Test your knowledge with AI-generated questions!")
            
#             if st.button("Generate New Question", key="gen_question"):
#                 question_data = st.session_state.ai_tutor.generate_quiz_question(
#                     st.session_state.current_topic,
#                     st.session_state.get('difficulty_level', 3)
#                 )
#                 st.session_state.current_question = question_data
            
#             if st.session_state.get('current_question'):
#                 question = st.session_state.current_question
                
#                 st.markdown(f"**Question:** {question['question']}")
                
#                 # Multiple choice options
#                 user_answer = st.radio("Choose your answer:", question['options'], key="quiz_answer")
                
#                 if st.button("Submit Answer", key="submit_answer"):
#                     correct_letter = question['correct_answer']
#                     user_letter = user_answer[0] if user_answer else ""
#                     is_correct = user_letter == correct_letter
                    
#                     if is_correct:
#                         st.success("✅ Correct! " + question['explanation'])
#                         st.session_state.correct_answers = st.session_state.get('correct_answers', 0) + 1
#                         st.balloons()
#                     else:
#                         st.error(f"❌ Incorrect. The correct answer was {correct_letter}. " + question['explanation'])
                    
#                     st.session_state.questions_asked = st.session_state.get('questions_asked', 0) + 1
                    
#                     # Save quiz result
#                     save_conversation(
#                         st.session_state.session_id,
#                         f"Quiz Answer: {user_answer}",
#                         f"Correct: {is_correct}. {question['explanation']}",
#                         1.0 if is_correct else -0.5,
#                         'confident' if is_correct else 'confused',
#                         5.0,
#                         is_correct,
#                         st.session_state.get('difficulty_level', 3)
#                     )
        
#         with tab3:
#             st.markdown("### 🎴 Your Flashcards")
#             st.markdown("Review key concepts with AI-generated flashcards!")
            
#             # Generate flashcards from recent conversations
#             if st.button("Generate Flashcards from Recent Learning", key="gen_flashcards"):
#                 # Get recent conversation content
#                 recent_content = " ".join([msg["content"] for msg in st.session_state.get('chat_history', [])[-10:]])
                
#                 flashcards = FlashcardGenerator.generate_flashcards(
#                     st.session_state.current_topic,
#                     recent_content,
#                     st.session_state.get('difficulty_level', 3)
#                 )
                
#                 st.session_state.flashcards = flashcards
            
#             # Display flashcards
#             if st.session_state.get('flashcards'):
#                 for i, card in enumerate(st.session_state.flashcards):
#                     with st.expander(f"Flashcard {i+1}: {card['question'][:50]}..."):
#                         st.markdown(f"**Question:** {card['question']}")
                        
#                         if st.button(f"Show Answer", key=f"show_answer_{i}"):
#                             st.markdown(f"**Answer:** {card['answer']}")
                            
#                             col1, col2, col3 = st.columns(3)
#                             with col1:
#                                 if st.button("😊 Easy", key=f"easy_{i}"):
#                                     st.success("Marked as easy!")
#                             with col2:
#                                 if st.button("🤔 Medium", key=f"medium_{i}"):
#                                     st.info("Marked as medium!")
#                             with col3:
#                                 if st.button("😓 Hard", key=f"hard_{i}"):
#                                     st.warning("Marked as hard!")
        
#         with tab4:
#             st.markdown("### 📊 Learning Analytics")
            
#             # Real-time learning progress
#             col1, col2, col3, col4 = st.columns(4)
            
#             with col1:
#                 st.metric(
#                     "Current Difficulty", 
#                     f"{st.session_state.get('difficulty_level', 3)}/10",
#                     delta=st.session_state.get('difficulty_level', 3) - 3
#                 )
            
#             with col2:
#                 questions_asked = st.session_state.get('questions_asked', 0)
#                 st.metric("Questions Answered", questions_asked)
            
#             with col3:
#                 correct_answers = st.session_state.get('correct_answers', 0)
#                 accuracy = (correct_answers / max(questions_asked, 1)) * 100
#                 st.metric("Session Accuracy", f"{accuracy:.1f}%")
            
#             with col4:
#                 chat_messages = len([msg for msg in st.session_state.get('chat_history', []) if msg['role'] == 'user'])
#                 st.metric("Messages Sent", chat_messages)
            
#             # Difficulty progression chart
#             if st.session_state.get('chat_history'):
#                 difficulty_data = []
#                 for i, msg in enumerate(st.session_state.get('chat_history', [])):
#                     if msg['role'] == 'user':
#                         difficulty_data.append({
#                             'Message': i//2 + 1,
#                             'Difficulty': st.session_state.get('difficulty_level', 3),
#                             'Emotion': msg.get('emotion', 'neutral')
#                         })
                
#                 if difficulty_data:
#                     df = pd.DataFrame(difficulty_data)
                    
#                     # Difficulty progression line chart
#                     fig = px.line(df, x='Message', y='Difficulty', 
#                                  title='Difficulty Adjustment Over Time',
#                                  color='Emotion')
#                     fig.update_layout(height=400)
#                     st.plotly_chart(fig, use_container_width=True)
                    
#                     # Emotion distribution
#                     emotion_counts = df['Emotion'].value_counts()
#                     fig2 = px.bar(x=emotion_counts.index, y=emotion_counts.values,
#                                  title='Emotions During This Session',
#                                  color=emotion_counts.index)
#                     fig2.update_layout(height=300)
#                     st.plotly_chart(fig2, use_container_width=True)
    
#     else:
#         # Landing page for users without setup
#         st.markdown("""
#         ## 🌟 Welcome to AI Mentor!
        
#         **Your Personalized Learning Coach with Emotion & Performance Awareness**
        
#         ### ✨ Features:
#         - 🤖 **Conversational AI Tutor** - Chat naturally with your AI mentor
#         - 😊 **Emotion Detection** - AI adapts to your learning mood
#         - 📈 **Performance Tracking** - Dynamic difficulty adjustment
#         - 📝 **Smart Quizzes** - AI-generated questions at your level
#         - 🎴 **Auto Flashcards** - Review cards created from your learning
#         - 📊 **Learning Analytics** - Track your progress and patterns
        
#         ### 🚀 Getting Started:
#         1. Enter your username in the sidebar
#         2. Add your OpenAI API key (get one at openai.com)
#         3. Choose a topic to start learning
#         4. Begin chatting with your AI mentor!
        
#         ### 💡 How It Works:
#         Your AI mentor analyzes your responses to detect emotions like confusion, confidence, or frustration. 
#         It then adapts the teaching style, difficulty level, and content to match your current state and learning pace.
#         """)
        
#         # Demo video or screenshots could go here
#         st.info("💡 **Pro Tip**: The AI mentor works best when you're honest about your understanding. Don't hesitate to say when you're confused!")

# if __name__ == "__main__":
#     main()


import streamlit as st
import json
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sqlite3
import hashlib
from textblob import TextBlob
import random
import re
from typing import Dict, List, Tuple
import requests

# Configure Streamlit page
st.set_page_config(
    page_title="AI Mentor - Personalized Learning Coach",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        padding: 1rem;
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .user-message {
        background-color: #E3F2FD;
        border-left: 4px solid #2196F3;
    }
    .ai-message {
        background-color: #F3E5F5;
        border-left: 4px solid #9C27B0;
    }
    .emotion-indicator {
        padding: 0.5rem;
        border-radius: 20px;
        text-align: center;
        margin: 0.5rem 0;
        font-weight: bold;
    }
    .confident { background-color: #C8E6C9; color: #2E7D32; }
    .confused { background-color: #FFECB3; color: #F57C00; }
    .frustrated { background-color: #FFCDD2; color: #C62828; }
    .excited { background-color: #E1BEE7; color: #7B1FA2; }
    .neutral { background-color: #F5F5F5; color: #424242; }
    .progress-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .flashcard {
        background-color: #FFF3E0;
        border: 2px solid #FF9800;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        cursor: pointer;
        transition: transform 0.2s;
    }
    .flashcard:hover {
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# Database setup
def init_database():
    conn = sqlite3.connect('ai_mentor.db')
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE,
            password_hash TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Learning sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            topic TEXT,
            start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            end_time TIMESTAMP,
            total_questions INTEGER DEFAULT 0,
            correct_answers INTEGER DEFAULT 0,
            avg_sentiment REAL DEFAULT 0.0,
            difficulty_level INTEGER DEFAULT 3,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Conversations table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY,
            session_id INTEGER,
            user_message TEXT,
            ai_response TEXT,
            sentiment_score REAL,
            emotion_detected TEXT,
            response_time REAL,
            is_correct BOOLEAN,
            difficulty_level INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions (id)
        )
    ''')
    
    # Flashcards table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS flashcards (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            topic TEXT,
            question TEXT,
            answer TEXT,
            difficulty INTEGER,
            times_reviewed INTEGER DEFAULT 0,
            last_reviewed TIMESTAMP,
            mastery_level REAL DEFAULT 0.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

# Sentiment Analysis Engine
class SentimentAnalyzer:
    def __init__(self):
        self.emotion_keywords = {
            'confused': ['confused', 'lost', 'don\'t understand', 'unclear', 'what', 'huh', '?', 'help'],
            'frustrated': ['frustrated', 'annoying', 'difficult', 'hard', 'stuck', 'can\'t', 'impossible'],
            'confident': ['yes', 'got it', 'understand', 'clear', 'easy', 'sure', 'know'],
            'excited': ['cool', 'awesome', 'interesting', 'wow', 'amazing', 'love', 'great']
        }
    
    def analyze_emotion(self, text: str, response_time: float = None, is_correct: bool = None) -> Dict:
        # Text-based sentiment analysis
        blob = TextBlob(text.lower())
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Keyword-based emotion detection
        emotion_scores = {}
        text_lower = text.lower()
        for emotion, keywords in self.emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            emotion_scores[emotion] = score
        
        # Determine primary emotion
        primary_emotion = max(emotion_scores, key=emotion_scores.get) if max(emotion_scores.values()) > 0 else 'neutral'
        
        # Adjust based on correctness and response time
        if is_correct is not None:
            if is_correct and primary_emotion == 'neutral':
                primary_emotion = 'confident'
            elif not is_correct and primary_emotion == 'neutral':
                primary_emotion = 'confused'
        
        if response_time and response_time > 30:  # Long response time might indicate confusion
            if primary_emotion in ['neutral', 'confident']:
                primary_emotion = 'confused'
        
        return {
            'emotion': primary_emotion,
            'polarity': polarity,
            'subjectivity': subjectivity,
            'confidence': max(emotion_scores.values()) / len(text.split()) if text else 0
        }

# Adaptive Difficulty Engine
class DifficultyAdapter:
    def __init__(self):
        self.min_difficulty = 1
        self.max_difficulty = 10
        self.adjustment_factors = {
            'confident': 0.5,
            'excited': 0.3,
            'confused': -0.7,
            'frustrated': -1.0,
            'neutral': 0.0
        }
    
    def adjust_difficulty(self, current_difficulty: int, emotion: str, accuracy: float, response_time: float) -> int:
        adjustment = 0
        
        # Emotion-based adjustment
        adjustment += self.adjustment_factors.get(emotion, 0)
        
        # Performance-based adjustment
        if accuracy > 0.8:
            adjustment += 0.3
        elif accuracy < 0.5:
            adjustment -= 0.5
        
        # Response time adjustment
        if response_time < 10:  # Quick response
            adjustment += 0.2
        elif response_time > 45:  # Slow response
            adjustment -= 0.3
        
        new_difficulty = current_difficulty + adjustment
        return max(self.min_difficulty, min(self.max_difficulty, round(new_difficulty)))

# AI Tutor Engine (Offline fallback)
class AITutor:
    def __init__(self):
        self.conversation_history = []
        self.current_topic = None
        self.difficulty_level = 3
        self.knowledge_base = {
            "Newton's Laws of Physics": {
                "concepts": ["inertia", "force", "acceleration", "action-reaction", "momentum"],
                "explanations": {
                    "inertia": "An object at rest stays at rest, an object in motion stays in motion unless acted upon by a force.",
                    "force": "Force equals mass times acceleration (F = ma).",
                    "action-reaction": "For every action, there is an equal and opposite reaction."
                }
            },
            "Algebra Fundamentals": {
                "concepts": ["variables", "equations", "functions", "polynomials", "factoring"],
                "explanations": {
                    "variables": "Variables are symbols that represent unknown numbers.",
                    "equations": "An equation states that two expressions are equal.",
                    "functions": "A function is a relation where each input has exactly one output."
                }
            },
            "Basic Chemistry": {
                "concepts": ["atoms", "molecules", "elements", "compounds", "reactions"],
                "explanations": {
                    "atoms": "Atoms are the basic building blocks of matter.",
                    "molecules": "Molecules are groups of atoms bonded together.",
                    "elements": "Elements are pure substances made of only one type of atom."
                }
            }
        }
        
    def set_topic(self, topic: str):
        self.current_topic = topic
        self.conversation_history = []
    
    def generate_response(self, user_input: str, emotion: str, context: str = "") -> str:
        topic_data = self.knowledge_base.get(self.current_topic, {})
        concepts = topic_data.get("concepts", [])
        explanations = topic_data.get("explanations", {})
        
        # Simple keyword matching for responses
        user_lower = user_input.lower()
        
        # Emotion-based response adaptation
        if emotion == 'confused':
            response_start = "I understand this can be confusing. Let me explain it more simply: "
        elif emotion == 'frustrated':
            response_start = "Don't worry, this is challenging for many students. Let's break it down: "
        elif emotion == 'excited':
            response_start = "Great enthusiasm! Let me share something interesting: "
        elif emotion == 'confident':
            response_start = "You're doing well! Here's some additional information: "
        else:
            response_start = "Let me help you with that: "
        
        # Find relevant concept
        relevant_concept = None
        for concept in concepts:
            if concept in user_lower:
                relevant_concept = concept
                break
        
        if relevant_concept and relevant_concept in explanations:
            explanation = explanations[relevant_concept]
            return f"{response_start}{explanation}"
        
        # Generic responses based on common question patterns
        if any(word in user_lower for word in ['what', 'define', 'explain']):
            if concepts:
                concept = random.choice(concepts)
                if concept in explanations:
                    return f"{response_start}Let's talk about {concept}. {explanations[concept]}"
        
        if any(word in user_lower for word in ['how', 'why']):
            return f"{response_start}This is a great question about {self.current_topic}. The key concept here involves understanding the fundamental principles and applying them step by step."
        
        if any(word in user_lower for word in ['example', 'show me']):
            return f"{response_start}Here's a practical example related to {self.current_topic}: Consider a real-world scenario where these principles apply directly."
        
        # Default response
        return f"{response_start}That's an interesting question about {self.current_topic}. Could you be more specific about what aspect you'd like to explore?"
    
    def generate_quiz_question(self, topic: str, difficulty: int) -> Dict:
        topic_data = self.knowledge_base.get(topic, {})
        concepts = topic_data.get("concepts", ["general concept"])
        
        if topic == "Newton's Laws of Physics":
            questions = [
                {
                    "question": "Which law states that an object at rest stays at rest?",
                    "options": ["A) First Law", "B) Second Law", "C) Third Law", "D) Fourth Law"],
                    "correct_answer": "A",
                    "explanation": "Newton's First Law (Law of Inertia) states that objects at rest stay at rest."
                },
                {
                    "question": "What is the formula for Newton's Second Law?",
                    "options": ["A) F = m/a", "B) F = ma", "C) F = a/m", "D) F = m + a"],
                    "correct_answer": "B",
                    "explanation": "Newton's Second Law is expressed as F = ma (Force = mass × acceleration)."
                }
            ]
        elif topic == "Algebra Fundamentals":
            questions = [
                {
                    "question": "What is a variable in algebra?",
                    "options": ["A) A number", "B) A symbol for unknown", "C) An equation", "D) A constant"],
                    "correct_answer": "B",
                    "explanation": "A variable is a symbol (like x or y) that represents an unknown number."
                },
                {
                    "question": "If 2x + 5 = 11, what is x?",
                    "options": ["A) 2", "B) 3", "C) 4", "D) 5"],
                    "correct_answer": "B",
                    "explanation": "Solving: 2x + 5 = 11, so 2x = 6, therefore x = 3."
                }
            ]
        else:
            questions = [
                {
                    "question": f"What is a key concept in {topic}?",
                    "options": ["A) Concept A", "B) Concept B", "C) Concept C", "D) Concept D"],
                    "correct_answer": "A",
                    "explanation": "This is one of the fundamental concepts in the subject."
                }
            ]
        
        return random.choice(questions)

# Flashcard Generator
class FlashcardGenerator:
    @staticmethod
    def generate_flashcards(topic: str, conversation_content: str, difficulty: int) -> List[Dict]:
        # Enhanced keyword extraction for flashcard generation
        keywords = re.findall(r'\b[A-Z][a-z]{3,}\b', conversation_content)
        # Also extract important terms mentioned in conversation
        important_terms = re.findall(r'\b(?:define|what is|explain|concept|principle|law|formula|equation)\s+([A-Za-z\s]{3,15})', conversation_content.lower())
        
        all_terms = list(set(keywords + [term.strip() for term in important_terms]))[:5]
        
        flashcards = []
        for i, term in enumerate(all_terms):
            if term:
                flashcard = {
                    'question': f"What is {term} in the context of {topic}?",
                    'answer': f"A key concept in {topic} related to the fundamental principles and applications we've discussed.",
                    'difficulty': difficulty,
                    'topic': topic,
                    'id': i
                }
                flashcards.append(flashcard)
        
        # Add some default flashcards if none were generated
        if not flashcards:
            default_cards = [
                {
                    'question': f"What are the main principles of {topic}?",
                    'answer': f"The fundamental concepts that govern {topic} and their practical applications.",
                    'difficulty': difficulty,
                    'topic': topic,
                    'id': 0
                },
                {
                    'question': f"How is {topic} applied in real life?",
                    'answer': f"Practical applications and examples of {topic} in everyday situations.",
                    'difficulty': difficulty,
                    'topic': topic,
                    'id': 1
                }
            ]
            flashcards = default_cards
        
        return flashcards

# Database Helper Functions
def get_user_id(username: str) -> int:
    conn = sqlite3.connect('ai_mentor.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
    result = cursor.fetchone()
    
    if result:
        user_id = result[0]
    else:
        cursor.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', 
                      (username, hash_password("default")))
        user_id = cursor.lastrowid
        conn.commit()
    
    conn.close()
    return user_id

def create_session(user_id: int, topic: str) -> int:
    conn = sqlite3.connect('ai_mentor.db')
    cursor = conn.cursor()
    
    cursor.execute('INSERT INTO sessions (user_id, topic) VALUES (?, ?)', (user_id, topic))
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return session_id

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_user(username: str, password: str) -> bool:
    conn = sqlite3.connect('ai_mentor.db')
    cursor = conn.cursor()
    cursor.execute('SELECT password_hash FROM users WHERE username = ?', (username,))
    result = cursor.fetchone()
    conn.close()
    if result:
        return result[0] == hash_password(password)
    return False

def register_user(username: str, password: str) -> bool:
    conn = sqlite3.connect('ai_mentor.db')
    cursor = conn.cursor()
    try:
        cursor.execute(
            'INSERT INTO users (username, password_hash) VALUES (?, ?)',
            (username, hash_password(password))
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def save_conversation(session_id: int, user_message: str, ai_response: str, 
                     sentiment_score: float, emotion: str, response_time: float,
                     is_correct: bool = None, difficulty: int = 3):
    conn = sqlite3.connect('ai_mentor.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO conversations 
        (session_id, user_message, ai_response, sentiment_score, emotion_detected, 
         response_time, is_correct, difficulty_level)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (session_id, user_message, ai_response, sentiment_score, emotion, 
          response_time, is_correct, difficulty))
    
    conn.commit()
    conn.close()

def get_user_analytics(user_id: int) -> Dict:
    conn = sqlite3.connect('ai_mentor.db')
    cursor = conn.cursor()
    
    # Get session stats
    cursor.execute('''
        SELECT COUNT(*) as total_sessions,
               AVG(CASE WHEN total_questions > 0 THEN correct_answers * 1.0 / total_questions ELSE 0 END) as avg_accuracy,
               AVG(avg_sentiment) as avg_sentiment
        FROM sessions WHERE user_id = ?
    ''', (user_id,))
    
    session_stats = cursor.fetchone()
    
    # Get emotion distribution
    cursor.execute('''
        SELECT emotion_detected, COUNT(*) as count
        FROM conversations c
        JOIN sessions s ON c.session_id = s.id
        WHERE s.user_id = ?
        GROUP BY emotion_detected
    ''', (user_id,))
    
    emotion_data = cursor.fetchall()
    
    conn.close()
    
    return {
        'total_sessions': session_stats[0] or 0,
        'avg_accuracy': session_stats[1] or 0,
        'avg_sentiment': session_stats[2] or 0,
        'emotion_distribution': dict(emotion_data) if emotion_data else {}
    }

# Initialize components
@st.cache_resource
def initialize_components():
    init_database()
    sentiment_analyzer = SentimentAnalyzer()
    difficulty_adapter = DifficultyAdapter()
    return sentiment_analyzer, difficulty_adapter
import os
# Main Application
def main():
           # Add this at the beginning of main() to reset database
# import os
#  if st.button("Reset Database (Development Only)", key="reset_db"):
#     if os.path.exists('ai_mentor.db'):
#         os.remove('ai_mentor.db')
#     st.success("Database reset! Please refresh the page.")
    # st.stop()
    st.markdown('<h1 class="main-header">🤖 AI Mentor</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Your Personalized Learning Coach with Emotion & Performance Awareness</p>', unsafe_allow_html=True)
    
    # Initialize components
    sentiment_analyzer, difficulty_adapter = initialize_components()
    
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    # Authentication
    if not st.session_state.logged_in:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("### 🔐 Welcome! Please Login or Register")
            
            tab1, tab2 = st.tabs(["Login", "Register"])
            
            with tab1:
                with st.form("login_form"):
                    username = st.text_input("Username")
                    password = st.text_input("Password", type="password")
                    login_btn = st.form_submit_button("Login")
                    
                    if login_btn:
                        if username and password:
                            if verify_user(username, password):
                                st.session_state.logged_in = True
                                st.session_state.username = username
                                st.success("✅ Logged in successfully!")
                                st.rerun()
                            else:
                                st.error("❌ Invalid credentials.")
                        else:
                            st.warning("Please enter both username and password.")
            
            with tab2:
                with st.form("register_form"):
                    new_username = st.text_input("New Username")
                    new_password = st.text_input("New Password", type="password")
                    confirm_password = st.text_input("Confirm Password", type="password")
                    register_btn = st.form_submit_button("Register")
                    
                    if register_btn:
                        if new_username and new_password and confirm_password:
                            if new_password == confirm_password:
                                if register_user(new_username, new_password):
                                    st.success("✅ Registered successfully! Please login.")
                                else:
                                    st.error("❌ Username already exists.")
                            else:
                                st.error("❌ Passwords don't match.")
                        else:
                            st.warning("Please fill all fields.")
            
            # Demo login option
            st.markdown("---")
            if st.button("🎯 Try Demo (No Registration Required)"):
                st.session_state.logged_in = True
                st.session_state.username = "demo_user"
                st.rerun()
        
        return
    
    # Sidebar for navigation and settings
    with st.sidebar:
        st.header("🎯 Learning Dashboard")
        st.success(f"Welcome, {st.session_state.username}! 👋")
        
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.clear()
            st.rerun()
        
        user_id = get_user_id(st.session_state.username)
        
        # Topic selection
        topics = [
            "Newton's Laws of Physics",
            "Algebra Fundamentals", 
            "Basic Chemistry",
            "Biology Cells",
            "Calculus Derivatives",
            "Statistics & Probability",
            "Computer Science Algorithms",
            "Geometry Theorems"
        ]
        
        selected_topic = st.selectbox("Choose a topic to learn:", topics)
        
        # Initialize AI tutor
        if 'ai_tutor' not in st.session_state or st.session_state.get('current_topic') != selected_topic:
            st.session_state.ai_tutor = AITutor()
            st.session_state.ai_tutor.set_topic(selected_topic)
            st.session_state.current_topic = selected_topic
            st.session_state.session_id = create_session(user_id, selected_topic)
            st.session_state.difficulty_level = 3
            st.session_state.questions_asked = 0
            st.session_state.correct_answers = 0
        
        # Display current stats
        st.markdown("### 📊 Current Session")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Difficulty", f"{st.session_state.get('difficulty_level', 3)}/10")
        with col2:
            accuracy = (st.session_state.get('correct_answers', 0) / max(st.session_state.get('questions_asked', 1), 1)) * 100
            st.metric("Accuracy", f"{accuracy:.1f}%")
        
        # User analytics
        analytics = get_user_analytics(user_id)
        st.markdown("### 📈 Your Progress")
        st.metric("Total Sessions", analytics['total_sessions'])
        st.metric("Overall Accuracy", f"{analytics['avg_accuracy']*100:.1f}%")
        
        # Emotion distribution chart
        if analytics['emotion_distribution']:
            fig = px.pie(
                values=list(analytics['emotion_distribution'].values()),
                names=list(analytics['emotion_distribution'].keys()),
                title="Your Learning Emotions"
            )
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # Main content area
    if st.session_state.get('ai_tutor'):
        # Create tabs for different features
        tab1, tab2, tab3, tab4 = st.tabs(["💬 AI Tutor Chat", "📝 Practice Quiz", "🎴 Flashcards", "📊 Analytics"])
        
        with tab1:
            st.markdown("### 🤖 Chat with Your AI Mentor")
            
            # Initialize chat history
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
                # Welcome message
                welcome_msg = f"Hello! I'm your AI mentor for {st.session_state.current_topic}. I'm here to help you learn at your own pace. What would you like to know?"
                st.session_state.chat_history.append({"role": "ai", "content": welcome_msg, "emotion": "neutral"})
            
            # Display chat history
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    with st.container():
                        st.markdown(f'<div class="chat-message user-message">{message["content"]}</div>', unsafe_allow_html=True)
                        if "emotion" in message:
                            emotion_class = message["emotion"]
                            st.markdown(f'<div class="emotion-indicator {emotion_class}">Detected: {emotion_class.title()}</div>', unsafe_allow_html=True)
                else:
                    with st.container():
                        st.markdown(f'<div class="chat-message ai-message">{message["content"]}</div>', unsafe_allow_html=True)
            
            # Chat input
            with st.form("chat_form"):
                user_input = st.text_input("Ask me anything about the topic:", key="chat_input")
                send_button = st.form_submit_button("Send")
            
            if send_button and user_input:
                start_time = time.time()
                
                # Analyze sentiment
                emotion_analysis = sentiment_analyzer.analyze_emotion(user_input)
                emotion = emotion_analysis['emotion']
                sentiment_score = emotion_analysis['polarity']
                
                response_time = time.time() - start_time
                
                # Generate AI response
                ai_response = st.session_state.ai_tutor.generate_response(
                    user_input, emotion, context=st.session_state.current_topic
                )
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "role": "user", 
                    "content": user_input, 
                    "emotion": emotion
                })
                st.session_state.chat_history.append({
                    "role": "ai", 
                    "content": ai_response
                })
                
                # Save to database
                save_conversation(
                    st.session_state.session_id,
                    user_input,
                    ai_response,
                    sentiment_score,
                    emotion,
                    response_time
                )
                
                # Adjust difficulty based on emotion
                current_accuracy = st.session_state.get('correct_answers', 0) / max(st.session_state.get('questions_asked', 1), 1)
                new_difficulty = difficulty_adapter.adjust_difficulty(
                    st.session_state.get('difficulty_level', 3),
                    emotion,
                    current_accuracy,
                    response_time
                )
                st.session_state.difficulty_level = new_difficulty
                st.session_state.ai_tutor.difficulty_level = new_difficulty
                
                st.rerun()
        
        with tab2:
            st.markdown("### 📝 Practice Quiz")
            st.markdown("Test your knowledge with AI-generated questions!")
            
            if st.button("Generate New Question", key="gen_question"):
                question_data = st.session_state.ai_tutor.generate_quiz_question(
                    st.session_state.current_topic,
                    st.session_state.get('difficulty_level', 3)
                )
                st.session_state.current_question = question_data
            
            if st.session_state.get('current_question'):
                question = st.session_state.current_question
                
                st.markdown(f"**Question:** {question['question']}")
                
                # Multiple choice options
                with st.form("quiz_form"):
                    user_answer = st.radio("Choose your answer:", question['options'], key="quiz_answer")
                    submit_answer = st.form_submit_button("Submit Answer")
                
                if submit_answer and user_answer:
                    correct_letter = question['correct_answer']
                    user_letter = user_answer[0] if user_answer else ""
                    is_correct = user_letter == correct_letter
                    # Update session stats
                    st.session_state.questions_asked = st.session_state.get('questions_asked', 0) + 1
                    if is_correct:
                        st.session_state.correct_answers = st.session_state.get('correct_answers', 0) + 1
                        st.success("✅ Correct! " + question['explanation'])
                        emotion = 'confident'
                    else:
                        st.error(f"❌ Incorrect. The correct answer is {correct_letter}. " + question['explanation'])
                        emotion = 'confused'
                    
                    # Save quiz result to database
                    save_conversation(
                        st.session_state.session_id,
                        f"Quiz: {question['question']} | Answer: {user_answer}",
                        f"Correct: {is_correct} | {question['explanation']}",
                        1.0 if is_correct else -0.5,
                        emotion,
                        2.0,  # Standard quiz response time
                        is_correct,
                        st.session_state.get('difficulty_level', 3)
                    )
                    
                    # Clear current question
                    st.session_state.current_question = None
                    
                    # Update difficulty based on performance
                    current_accuracy = st.session_state.get('correct_answers', 0) / st.session_state.get('questions_asked', 1)
                    new_difficulty = difficulty_adapter.adjust_difficulty(
                        st.session_state.get('difficulty_level', 3),
                        emotion,
                        current_accuracy,
                        2.0
                    )
                    st.session_state.difficulty_level = new_difficulty
                    
                    st.rerun()
        
        with tab3:
            st.markdown("### 🎴 AI-Generated Flashcards")
            st.markdown("Review key concepts with personalized flashcards!")
            
            # Generate flashcards button
            if st.button("Generate Flashcards from Chat", key="gen_flashcards"):
                # Get conversation content for flashcard generation
                chat_content = " ".join([msg["content"] for msg in st.session_state.get('chat_history', [])])
                
                if chat_content:
                    flashcards = FlashcardGenerator.generate_flashcards(
                        st.session_state.current_topic,
                        chat_content,
                        st.session_state.get('difficulty_level', 3)
                    )
                    st.session_state.flashcards = flashcards
                    
                    # Save flashcards to database
                    conn = sqlite3.connect('ai_mentor.db')
                    cursor = conn.cursor()
                    
                    for card in flashcards:
                        cursor.execute('''
                            INSERT INTO flashcards 
                            (user_id, topic, question, answer, difficulty, mastery_level)
                            VALUES (?, ?, ?, ?, ?, ?)
                        ''', (user_id, card['topic'], card['question'], card['answer'], 
                              card['difficulty'], 0.0))
                    
                    conn.commit()
                    conn.close()
                    
                    st.success(f"Generated {len(flashcards)} flashcards!")
                else:
                    st.warning("Start chatting first to generate personalized flashcards!")
            
            # Display flashcards
            if st.session_state.get('flashcards'):
                if 'current_card_index' not in st.session_state:
                    st.session_state.current_card_index = 0
                    st.session_state.show_answer = False
                
                card = st.session_state.flashcards[st.session_state.current_card_index]
                total_cards = len(st.session_state.flashcards)
                
                st.markdown(f"**Card {st.session_state.current_card_index + 1} of {total_cards}**")
                
                # Flashcard display
                with st.container():
                    if not st.session_state.get('show_answer', False):
                        st.markdown(f'''
                        <div class="flashcard">
                            <h3>Question:</h3>
                            <p style="font-size: 1.2rem;">{card['question']}</p>
                            <p style="text-align: center; color: #666;">Click "Show Answer" to reveal</p>
                        </div>
                        ''', unsafe_allow_html=True)
                    else:
                        st.markdown(f'''
                        <div class="flashcard">
                            <h3>Question:</h3>
                            <p>{card['question']}</p>
                            <h3>Answer:</h3>
                            <p style="font-size: 1.1rem; color: #2E86AB;">{card['answer']}</p>
                        </div>
                        ''', unsafe_allow_html=True)
                
                # Flashcard controls
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if st.button("⬅️ Previous"):
                        st.session_state.current_card_index = (st.session_state.current_card_index - 1) % total_cards
                        st.session_state.show_answer = False
                        st.rerun()
                
                with col2:
                    if st.button("Show Answer" if not st.session_state.get('show_answer') else "Hide Answer"):
                        st.session_state.show_answer = not st.session_state.get('show_answer', False)
                        st.rerun()
                
                with col3:
                    if st.button("➡️ Next"):
                        st.session_state.current_card_index = (st.session_state.current_card_index + 1) % total_cards
                        st.session_state.show_answer = False
                        st.rerun()
                
                with col4:
                    if st.button("🔀 Shuffle"):
                        random.shuffle(st.session_state.flashcards)
                        st.session_state.current_card_index = 0
                        st.session_state.show_answer = False
                        st.rerun()
                
                # Self-assessment
                if st.session_state.get('show_answer'):
                    st.markdown("**How well did you know this?**")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("😰 Didn't Know", key="dont_know"):
                            st.info("Marked as needs review")
                            st.session_state.current_card_index = (st.session_state.current_card_index + 1) % total_cards
                            st.session_state.show_answer = False
                            st.rerun()
                    
                    with col2:
                        if st.button("🤔 Somewhat", key="somewhat"):
                            st.info("Marked for more practice")
                            st.session_state.current_card_index = (st.session_state.current_card_index + 1) % total_cards
                            st.session_state.show_answer = False
                            st.rerun()
                    
                    with col3:
                        if st.button("✅ Knew It!", key="knew_it"):
                            st.success("Great! Marked as mastered")
                            st.session_state.current_card_index = (st.session_state.current_card_index + 1) % total_cards
                            st.session_state.show_answer = False
                            st.rerun()
            
            # Load saved flashcards
            st.markdown("---")
            if st.button("Load My Saved Flashcards"):
                conn = sqlite3.connect('ai_mentor.db')
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT question, answer, difficulty, topic, times_reviewed, mastery_level
                    FROM flashcards 
                    WHERE user_id = ? 
                    ORDER BY created_at DESC 
                    LIMIT 20
                ''', (user_id,))
                
                saved_cards = cursor.fetchall()
                conn.close()
                
                if saved_cards:
                    flashcards = []
                    for i, card in enumerate(saved_cards):
                        flashcards.append({
                            'question': card[0],
                            'answer': card[1],
                            'difficulty': card[2],
                            'topic': card[3],
                            'id': i
                        })
                    st.session_state.flashcards = flashcards
                    st.session_state.current_card_index = 0
                    st.session_state.show_answer = False
                    st.success(f"Loaded {len(flashcards)} saved flashcards!")
                    st.rerun()
                else:
                    st.info("No saved flashcards found. Generate some first!")
        
        with tab4:
            st.markdown("### 📊 Learning Analytics")
            
            # Session performance
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('''
                <div class="progress-card">
                    <h3>Current Session</h3>
                    <p>Questions: {}</p>
                    <p>Correct: {}</p>
                    <p>Accuracy: {:.1f}%</p>
                </div>
                '''.format(
                    st.session_state.get('questions_asked', 0),
                    st.session_state.get('correct_answers', 0),
                    (st.session_state.get('correct_answers', 0) / max(st.session_state.get('questions_asked', 1), 1)) * 100
                ), unsafe_allow_html=True)
            
            with col2:
                st.markdown('''
                <div class="progress-card">
                    <h3>Difficulty Level</h3>
                    <p>Current: {}/10</p>
                    <p>Topic: {}</p>
                    <p>Adaptive: ✅</p>
                </div>
                '''.format(
                    st.session_state.get('difficulty_level', 3),
                    st.session_state.current_topic
                ), unsafe_allow_html=True)
            
            with col3:
                st.markdown('''
                <div class="progress-card">
                    <h3>Overall Stats</h3>
                    <p>Sessions: {}</p>
                    <p>Avg Accuracy: {:.1f}%</p>
                    <p>Sentiment: {:.2f}</p>
                </div>
                '''.format(
                    analytics['total_sessions'],
                    analytics['avg_accuracy'] * 100,
                    analytics['avg_sentiment']
                ), unsafe_allow_html=True)
            
            # Learning progress over time
            st.markdown("### 📈 Learning Progress")
            
            # Get session history
            conn = sqlite3.connect('ai_mentor.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT DATE(start_time) as date, 
                       AVG(CASE WHEN total_questions > 0 THEN correct_answers * 1.0 / total_questions ELSE 0 END) as accuracy,
                       COUNT(*) as sessions
                FROM sessions 
                WHERE user_id = ? AND start_time >= date('now', '-30 days')
                GROUP BY DATE(start_time)
                ORDER BY date
            ''', (user_id,))
            
            progress_data = cursor.fetchall()
            
            if progress_data:
                df = pd.DataFrame(progress_data, columns=['Date', 'Accuracy', 'Sessions'])
                
                # Accuracy over time
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['Date'],
                    y=df['Accuracy'] * 100,
                    mode='lines+markers',
                    name='Accuracy %',
                    line=dict(color='#2E86AB', width=3),
                    marker=dict(size=8)
                ))
                
                fig.update_layout(
                    title='Learning Accuracy Over Time',
                    xaxis_title='Date',
                    yaxis_title='Accuracy (%)',
                    hovermode='x',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Emotion trends
            cursor.execute('''
                SELECT emotion_detected, COUNT(*) as count,
                       DATE(timestamp) as date
                FROM conversations c
                JOIN sessions s ON c.session_id = s.id
                WHERE s.user_id = ? AND c.timestamp >= date('now', '-7 days')
                GROUP BY emotion_detected, DATE(timestamp)
                ORDER BY date DESC
            ''', (user_id,))
            
            emotion_trends = cursor.fetchall()
            conn.close()
            
            if emotion_trends:
                emotion_df = pd.DataFrame(emotion_trends, columns=['Emotion', 'Count', 'Date'])
                
                # Create emotion heatmap
                pivot_df = emotion_df.pivot_table(values='Count', index='Emotion', columns='Date', fill_value=0)
                
                fig = px.imshow(
                    pivot_df.values,
                    labels=dict(x="Date", y="Emotion", color="Frequency"),
                    x=pivot_df.columns,
                    y=pivot_df.index,
                    title="Emotion Patterns (Last 7 Days)"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Performance by topic
            st.markdown("### 📚 Performance by Topic")
            
            conn = sqlite3.connect('ai_mentor.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT topic, 
                       COUNT(*) as sessions,
                       AVG(CASE WHEN total_questions > 0 THEN correct_answers * 1.0 / total_questions ELSE 0 END) as avg_accuracy,
                       AVG(difficulty_level) as avg_difficulty
                FROM sessions 
                WHERE user_id = ?
                GROUP BY topic
                ORDER BY sessions DESC
            ''', (user_id,))
            
            topic_stats = cursor.fetchall()
            conn.close()
            
            if topic_stats:
                topic_df = pd.DataFrame(topic_stats, columns=['Topic', 'Sessions', 'Avg Accuracy', 'Avg Difficulty'])
                
                # Display as table
                st.dataframe(
                    topic_df.style.format({
                        'Avg Accuracy': '{:.1%}',
                        'Avg Difficulty': '{:.1f}'
                    }),
                    use_container_width=True
                )
                
                # Topic performance chart
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=topic_df['Topic'],
                    y=topic_df['Avg Accuracy'] * 100,
                    name='Accuracy %',
                    marker_color='lightblue'
                ))
                
                fig.update_layout(
                    title='Performance by Topic',
                    xaxis_title='Topic',
                    yaxis_title='Average Accuracy (%)',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Learning recommendations
            st.markdown("### 💡 Personalized Recommendations")
            
            recommendations = []
            
            # Based on accuracy
            current_accuracy = (st.session_state.get('correct_answers', 0) / 
                              max(st.session_state.get('questions_asked', 1), 1))
            
            if current_accuracy < 0.6:
                recommendations.append("📚 Consider reviewing the basics before moving to advanced topics")
            elif current_accuracy > 0.8:
                recommendations.append("🚀 You're doing great! Try increasing the difficulty level")
            
            # Based on emotions
            recent_emotions = [msg.get("emotion", "neutral") for msg in st.session_state.get('chat_history', []) 
                             if msg.get("role") == "user"][-5:]  # Last 5 user messages
            
            if recent_emotions.count('confused') > 2:
                recommendations.append("🤔 You seem confused lately. Try breaking down complex topics into smaller parts")
            elif recent_emotions.count('frustrated') > 1:
                recommendations.append("😤 Take a break! Sometimes stepping away helps consolidate learning")
            elif recent_emotions.count('excited') > 2:
                recommendations.append("🎉 Your enthusiasm is great! Channel it into exploring related topics")
            
            # Based on session length
            if st.session_state.get('questions_asked', 0) < 3:
                recommendations.append("⏰ Try longer study sessions for better retention")
            
            if not recommendations:
                recommendations.append("✨ Keep up the great work! Consistent practice leads to mastery")
            
            for rec in recommendations:
                st.info(rec)

# Run the application
if __name__ == "__main__":
    main()