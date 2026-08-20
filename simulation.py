import os
from typing import List, Dict, Tuple
from groq import Groq
from rag_engine import RAGEngine

# Persona Prompt Templates
PERSONAS = {
    "expert": {
        "name": "Prof. Ahmad Yani (Subject Matter Expert)",
        "role_desc": "Senior Professor who has published extensively in this domain. Focuses on theoretical foundation, novelty, related works, and contribution.",
        "avatar": "👨‍🏫",
        "system_prompt": """You are Prof. Ahmad Yani, a senior academic examiner. Your specialty is the theoretical domain and state-of-the-art literature related to this student's thesis.
Your tone is intellectual, formal, demanding, but constructive. You speak in a blend of academic Indonesian and English (which is standard for Indonesian Semhas panels).

Your goal: Verify if the student understands the literature, theoretical concepts, and if their work actually presents a novel contribution.

Instructions:
1. Examine the student's response and use the retrieved thesis context to formulate your next question.
2. Ask about the literature review, theoretical frameworks, or comparison with existing methods.
3. Be specific: cite terms or formulas mentioned in the retrieved context.
4. Keep your question concise, professional, and clear. Do not answer your own questions.
5. If the student's previous answer was weak, start with a brief critical comment on their answer before asking your next question.
"""
    },
    "methodologist": {
        "name": "Dr. Sarah Fitri (Methodology & Analytics)",
        "role_desc": "Associate Professor specializing in research design, statistics, data gathering, algorithms, and experimental validation.",
        "avatar": "👩‍🔬",
        "system_prompt": """You are Dr. Sarah Fitri, a sharp and detailed examiner. Your specialty is research methodology, experimental setups, validation metrics, datasets, and statistical logic.
Your tone is analytical, direct, and rigorous. You speak in a blend of academic Indonesian and English.

Your goal: Verify that the student's research design is statistically sound, valid, and that they executed their algorithms or models correctly.

Instructions:
1. Examine the student's response and use the retrieved thesis context to formulate your next question.
2. Ask about datasets (sizes, splitting, bias), evaluation metrics (precision, recall, F1, p-value), algorithms, hyperparameters, training processes, or equations.
3. Reference specific methodology points from the retrieved context.
4. Keep your question concise, challenging, and clear.
5. If they answered vaguely, push them on the details (e.g., "Anda menyebutkan akurasi tinggi, tapi berapa precision dan recall-nya secara spesifik?").
"""
    },
    "skeptic": {
        "name": "Dr. Edward Hutapea (The Critical Skeptic)",
        "role_desc": "External Examiner known for questioning practical viability, limits, failure cases, and asking the hard 'So What?' question.",
        "avatar": "🕵️‍♂️",
        "system_prompt": """You are Dr. Edward Hutapea, a skeptical examiner. Your focus is on practical implementation, cost, execution times, constraints, limitations, and real-world usefulness.
Your tone is critical, challenging, slightly intimidating, but professional. You speak in a blend of academic Indonesian and English.

Your goal: Test the student's critical thinking, how they handle stress, and see if they are aware of the limitations of their own work.

Instructions:
1. Examine the student's response and use the retrieved thesis context to formulate your next question.
2. Ask about constraints: Why is their approach viable? What are the limitations? What happens if it fails? What is the practical overhead? Ask them "So what? Why does this research matter?"
3. Challenge their findings and claims. Push them to defend their work against practical disadvantages.
4. Keep your question sharp, direct, and concise.
"""
    }
}

class SemhasSimulation:
    def __init__(self, rag_engine: RAGEngine, model_name: str = "qwen/qwen3.6-27b", api_key: str = None):
        self.rag = rag_engine
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        
        if not self.api_key:
            raise ValueError("Groq API Key is missing. Please set GROQ_API_KEY in your environment or .env file.")
            
        self.client = Groq(api_key=self.api_key)
        self.max_questions = 6 # 2 from each examiner
        self.reset_session()

    def reset_session(self):
        """Resets the simulation state to start a new interview."""
        self.chat_history: List[Dict[str, str]] = [] # list of {"role": "user/model", "content": "..."}
        self.question_count = 0
        self.current_examiner_key = "expert"
        self.last_retrieved_chunks: List[str] = []
        self.is_active = False

    def start_session(self, first_page_preview: str = "") -> str:
        """
        Starts the mock interview session. 
        Returns the initial greeting and the first question from the lead examiner (Prof. Expert).
        """
        self.reset_session()
        self.is_active = True
        self.question_count = 1
        
        prompt = f"""
        You are {PERSONAS['expert']['name']}, the chief examiner. The student is starting their Semhas (Seminar Hasil) presentation.
        We have processed their thesis abstract/first pages:
        ---
        {first_page_preview}
        ---
        Please open the session in a professional, realistic manner.
        1. Greet the student (Selamat siang / pagi).
        2. Briefly acknowledge the title/topic of their research.
        3. Ask them a standard opening question: ask them to summarize their main motivation, methods, and core contribution in 2-3 minutes.
        Speak in academic Indonesian mixed with technical English.
        """
        
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        first_question = completion.choices[0].message.content.strip()
        
        # Save to chat history
        self.chat_history.append({"role": "model", "content": f"[{PERSONAS['expert']['name']}]: {first_question}"})
        
        return first_question

    def _determine_next_examiner(self) -> str:
        """Rotates examiner keys based on the question count."""
        if self.question_count <= 2:
            return "expert"
        elif self.question_count <= 4:
            return "methodologist"
        else:
            return "skeptic"

    def submit_answer(self, student_answer: str) -> Dict:
        """
        Submits the student's answer, queries the RAG engine for relevant thesis context,
        determines the next examiner, and returns the next examiner's question.
        """
        if not self.is_active:
            raise ValueError("No active simulation session. Please start a session first.")
            
        # 1. Save student answer to history
        self.chat_history.append({"role": "user", "content": student_answer})
        
        # 2. Increment question count
        self.question_count += 1
        
        # Check if the interview is complete
        if self.question_count > self.max_questions:
            self.is_active = False
            return {
                "question": "Sesi tanya jawab telah selesai. Terima kasih atas jawaban Anda. Kami akan merundingkan nilai evaluasi Anda sekarang. Silakan klik tombol 'Dapatkan Evaluasi & Masukan' di tab Evaluasi.",
                "examiner": "Panel Penguji",
                "retrieved_chunks": [],
                "is_finished": True
            }
            
        # 3. Retrieve relevant context from thesis based on the answer and previous questions
        query = f"{self.chat_history[-2]['content']} {student_answer}"
        retrieved = self.rag.retrieve(query, top_k=3)
        self.last_retrieved_chunks = [item["content"] for item in retrieved]
        
        # Format the context for the LLM
        formatted_context = "\n---\n".join([f"Chunk {i+1}:\n{content}" for i, content in enumerate(self.last_retrieved_chunks)])
        
        # 4. Determine next examiner
        self.current_examiner_key = self._determine_next_examiner()
        next_persona = PERSONAS[self.current_examiner_key]
        
        # Build systematic chat message list for Groq API
        system_content = f"{next_persona['system_prompt']}\n\nRETRIEVED CONTEXT FROM STUDENT'S THESIS:\n{formatted_context}"
        
        messages = [
            {"role": "system", "content": system_content}
        ]
        
        # Build history matching assistant/user pairs
        for msg in self.chat_history:
            role = "assistant" if msg["role"] == "model" else "user"
            content = msg["content"]
            
            # Clean up examiner prefix for the assistant messages
            if role == "assistant" and content.startswith("["):
                close_idx = content.find("]")
                if close_idx != -1:
                    content = content[close_idx + 2:].strip()
            
            messages.append({"role": role, "content": content})
            
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.7,
        )
        next_question = completion.choices[0].message.content.strip()
        
        # Save question to history
        self.chat_history.append({"role": "model", "content": f"[{next_persona['name']}]: {next_question}"})
        
        return {
            "question": next_question,
            "examiner": next_persona["name"],
            "avatar": next_persona["avatar"],
            "retrieved_chunks": self.last_retrieved_chunks,
            "is_finished": False
        }

    def generate_evaluation(self) -> str:
        """
        Gathers the entire chat log, analyzes the student's defensive performance,
        and generates a detailed evaluation report with A-E grading and revision checklist.
        """
        if not self.chat_history:
            return "No interview session history found to evaluate."
            
        transcript = ""
        for msg in self.chat_history:
            role_label = "Student" if msg["role"] == "user" else "Examiner Panel"
            transcript += f"{role_label}: {msg['content']}\n\n"
            
        prompt = f"""
You are the Lead Examiner and Panel Chair of this Semhas (Seminar Hasil) thesis defense.
Analyze the following transcript of the defense:
---
{transcript}
---

Generate a comprehensive, formal **Evaluation Report** in Markdown format.

The report MUST contain:
1. **Grade & Verdict**: Assign a realistic GPA-style grade (e.g. 3.75 / A, 3.20 / B+, 2.50 / C) based on their answers. Be fair but strict. Indicate if they pass to the final defense (Sidang Akhir) or require major revisions.
2. **Scorecard (Out of 10)**:
   - *Methodology Defense* (Did they justify their algorithms, datasets, statistical testing?)
   - *Domain Knowledge* (Did they know the literature and theoretical concepts?)
   - *Clarity & Presentation* (Were their responses coherent and clear?)
   - *Critical & Analytical Thinking* (How well did they handle limitations and stress?)
3. **General Feedback**: Summarize their overall performance during the defense.
4. **Key Strengths**: 2-3 specific points where they excelled.
5. **Revisions Checklist (Critical & Recommended)**: A structured checklist pointing out exactly which sections of their thesis need improvements or clarifications, citing the weaknesses exposed in their mock answers.

Write the evaluation in professional academic Indonesian. Keep the output beautifully formatted with headers, lists, and tables.
"""
        
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
        )
        
        return completion.choices[0].message.content.strip()
