import os
import gradio as gr
from dotenv import load_dotenv
from rag_engine import RAGEngine
from simulation import SemhasSimulation, PERSONAS

# Load environment variables
load_dotenv()

# CSS for a premium, academic-defense look
CUSTOM_CSS = """
.container { max-width: 1200px; margin: 0 auto; }
.header { text-align: center; margin-bottom: 20px; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
.header h1 { margin: 0; font-size: 2.2em; font-weight: 700; }
.header p { margin: 5px 0 0 0; opacity: 0.9; }
.examiner-card { background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 15px; margin-bottom: 15px; }
.rag-box { background-color: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 15px; }
.rag-title { color: #166534; font-weight: 600; margin-bottom: 8px; font-size: 1.1em; display: flex; align-items: center; gap: 8px; }
.feedback-box { background-color: #faf5ff; border: 1px solid #e9d5ff; border-radius: 8px; padding: 20px; }
.btn-primary { background: #2563eb !important; color: white !important; }
.btn-success { background: #16a34a !important; color: white !important; }
"""

def process_file_ui(pdf_file, api_key):
    """Callback to process the uploaded PDF file."""
    if not pdf_file:
        return (
            gr.update(value="### Silakan unggah file PDF thesis Anda."),
            gr.update(visible=False),
            gr.update(interactive=False),
            None,
            None
        )
    
    # Use API key from UI textbox, fallback to env variable
    active_key = api_key.strip() if api_key else os.getenv("GROQ_API_KEY")
    if not active_key:
        return (
            gr.update(value="### ❌ Error: API Key tidak ditemukan. Silakan masukkan Groq API Key Anda."),
            gr.update(visible=False),
            gr.update(interactive=False),
            None,
            None
        )
        
    try:
        # Initialize engines
        rag_engine = RAGEngine(api_key=active_key)
        simulation = SemhasSimulation(rag_engine=rag_engine, api_key=active_key)
        
        # Parse PDF and build vector database
        abstract_preview, chunk_count = rag_engine.process_pdf(pdf_file.name)
        
        status_text = f"""
        ### 🟢 Dokumen Berhasil Diproses!
        *   **File**: `{os.path.basename(pdf_file.name)}`
        *   **Total Chunks**: `{chunk_count}`
        *   **Model Embedding**: `all-MiniLM-L6-v2 (Lokal Offline)`
        
        **Abstrak / Pendahuluan Preview**:
        {abstract_preview}
        """
        
        return (
            gr.update(value=status_text),
            gr.update(visible=True),
            gr.update(interactive=True),
            rag_engine,
            simulation
        )
    except Exception as e:
        return (
            gr.update(value=f"### ❌ Terjadi Kesalahan saat memproses PDF:\n`{str(e)}`"),
            gr.update(visible=False),
            gr.update(interactive=False),
            None,
            None
        )

def start_simulation_ui(simulation, rag):
    """Starts the simulation interview session."""
    if not simulation or not rag:
        return [], "", "", "", "", gr.update(interactive=False), gr.update(interactive=False)
        
    # Start the session
    first_page_preview = rag.chunks[0] if rag.chunks else "Thesis Paper"
    first_question = simulation.start_session(first_page_preview)
    
    # Format chatbot output
    # Gradio 6 Chatbot history format: list of dicts with role and content keys
    chatbot_history = [{"role": "assistant", "content": f"**[{PERSONAS['expert']['name']}]**\n{first_question}"}]
    
    # Active Examiner display info
    examiner_info = f"""
    ### Sedang Bertanya:
    ## {PERSONAS['expert']['avatar']} {PERSONAS['expert']['name']}
    *{PERSONAS['expert']['role_desc']}*
    """
    
    status_label = f"Pertanyaan 1 dari {simulation.max_questions}"
    
    return (
        chatbot_history,
        examiner_info,
        status_label,
        "", # Clear student input
        "Belum ada data retrieved untuk pertanyaan pembuka.",
        gr.update(interactive=True),  # Enable input text
        gr.update(interactive=False), # Disable eval button
        gr.update(visible=True)        # Show active question card
    )

def submit_answer_ui(student_answer, chatbot_history, simulation):
    """Submits student answer and gets the next question."""
    if not simulation:
        return chatbot_history, "", "", "", "", gr.update(interactive=False), gr.update(interactive=False)
        
    if not student_answer.strip():
        # Do not accept empty answers
        return chatbot_history, gr.update(), gr.update(), student_answer, gr.update(), gr.update(), gr.update()
        
    # 1. Update chatbot history to show user's response
    if chatbot_history is None:
        chatbot_history = []
    
    chatbot_history.append({"role": "user", "content": student_answer})
        
    # 2. Call simulation to get next response
    response_dict = simulation.submit_answer(student_answer)
    next_question = response_dict["question"]
    examiner_name = response_dict["examiner"]
    retrieved_chunks = response_dict["retrieved_chunks"]
    is_finished = response_dict["is_finished"]
    
    # Append the next examiner's question to the chatbot
    chatbot_history.append({"role": "assistant", "content": f"**[{examiner_name}]**\n{next_question}"})
    
    # Format retrieved RAG context for displaying in the side panel
    rag_context_md = "### 🔍 Retrieved Thesis Context (RAG):\n"
    if retrieved_chunks:
        for idx, chunk in enumerate(retrieved_chunks):
            rag_context_md += f"**Konteks {idx+1}:**\n> {chunk.strip()}\n\n---\n"
    else:
        rag_context_md += "*Tidak ada konteks yang diambil (Sesi Selesai/Pembukaan).*"
        
    # Prepare info of active examiner
    if not is_finished:
        current_key = simulation.current_examiner_key
        persona = PERSONAS[current_key]
        examiner_info = f"""
        ### Sedang Bertanya:
        ## {persona['avatar']} {persona['name']}
        *{persona['role_desc']}*
        """
        status_label = f"Pertanyaan {simulation.question_count} dari {simulation.max_questions}"
        input_interactive = gr.update(interactive=True)
        eval_interactive = gr.update(interactive=False)
    else:
        examiner_info = "### Sesi Tanya Jawab Selesai!\nSilakan beranjak ke tab Evaluasi."
        status_label = "Selesai"
        input_interactive = gr.update(interactive=False)
        eval_interactive = gr.update(interactive=True)
        
    return (
        chatbot_history,
        examiner_info,
        status_label,
        "", # Clear student input field
        rag_context_md,
        input_interactive,
        eval_interactive
    )

def generate_evaluation_ui(simulation):
    """Generates and returns the final scoring report."""
    if not simulation:
        return "### ❌ Silakan selesaikan ujian terlebih dahulu."
    
    # Trigger final evaluation from simulation
    report_md = simulation.generate_evaluation()
    return report_md

# Create the Gradio interface
with gr.Blocks(title="Semhas Mock Interview RAG") as demo:
    
    # State variables for persistence
    rag_state = gr.State()
    simulation_state = gr.State()
    
    # Header block
    with gr.Group(elem_classes="header"):
        gr.Markdown(
            """
            # 🎓 Semhas Mock Interview Simulator (RAG)
            ##### Uji pemahaman skripsi Anda dengan panel penguji bertenaga AI yang mempelajari dokumen Anda secara mendalam.
            """
        )
        
    with gr.Row(elem_classes="container"):
        
        # Left Panel (Settings & Processing)
        with gr.Column(scale=4):
            gr.Markdown("### 📂 Langkah 1: Unggah Dokumen Skripsi")
            
            # API Key input
            api_key_input = gr.Textbox(
                label="Groq API Key", 
                placeholder="Masukkan Groq API Key Anda (atau kosongkan jika sudah diset di .env)", 
                type="password"
            )
            
            # PDF File uploader
            file_uploader = gr.File(
                label="Unggah PDF Skripsi/Thesis (Format PDF)", 
                file_types=[".pdf"]
            )
            
            process_btn = gr.Button("Proses Dokumen & Bangun RAG", elem_classes="btn-primary")
            
            # Processing status report
            process_status = gr.Markdown("*Dokumen belum diunggah.*")
            
            # Start session button
            start_btn = gr.Button(
                "Mulai Simulasi Sidang Semhas", 
                elem_classes="btn-success", 
                interactive=False
            )
            
        # Right Panel (Interactive Defense Simulation)
        with gr.Column(scale=8):
            
            # Tabs for Simulation Console and Evaluation Report
            with gr.Tabs():
                
                with gr.Tab("💬 Simulasi Ujian"):
                    
                    with gr.Row(visible=False) as session_details_row:
                        current_examiner_card = gr.Markdown(elem_classes="examiner-card")
                        question_status_lbl = gr.Label(label="Status Pertanyaan")
                    
                    chatbot = gr.Chatbot(
                        label="Ruang Sidang Semhas", 
                        height=400
                    )
                    
                    with gr.Row():
                        student_input = gr.Textbox(
                            show_label=False, 
                            placeholder="Ketik jawaban sanggahan Anda di sini...",
                            interactive=False
                        )
                        submit_btn = gr.Button("Kirim Jawaban", scale=0)
                        
                    # Side block showing RAG Context
                    rag_context_panel = gr.Markdown(
                        value="*Mulai simulasi untuk melihat context yang dibaca RAG.*", 
                        elem_classes="rag-box"
                    )
                    
                with gr.Tab("📊 Hasil Evaluasi & Masukan"):
                    gr.Markdown("### Keputusan Panel Penguji")
                    
                    evaluate_btn = gr.Button(
                        "Dapatkan Evaluasi & Masukan", 
                        elem_classes="btn-primary", 
                        interactive=False
                    )
                    
                    evaluation_report = gr.Markdown(
                        value="*Hasil evaluasi akan muncul di sini setelah Anda menjawab seluruh pertanyaan penguji.*",
                        elem_classes="feedback-box"
                    )
                    
    # --- EVENT BINDINGS ---
    
    # 1. Processing files
    process_btn.click(
        fn=process_file_ui,
        inputs=[file_uploader, api_key_input],
        outputs=[process_status, start_btn, start_btn, rag_state, simulation_state]
    )
    
    # 2. Starting simulation
    start_btn.click(
        fn=start_simulation_ui,
        inputs=[simulation_state, rag_state],
        outputs=[
            chatbot, 
            current_examiner_card, 
            question_status_lbl, 
            student_input, 
            rag_context_panel, 
            student_input, 
            evaluate_btn,
            session_details_row
        ]
    )
    
    # 3. Submitting answers (via button click or enter key)
    submit_event = submit_btn.click(
        fn=submit_answer_ui,
        inputs=[student_input, chatbot, simulation_state],
        outputs=[chatbot, current_examiner_card, question_status_lbl, student_input, rag_context_panel, student_input, evaluate_btn]
    )
    
    student_input.submit(
        fn=submit_answer_ui,
        inputs=[student_input, chatbot, simulation_state],
        outputs=[chatbot, current_examiner_card, question_status_lbl, student_input, rag_context_panel, student_input, evaluate_btn]
    )
    
    # 4. Requesting evaluation
    evaluate_btn.click(
        fn=generate_evaluation_ui,
        inputs=[simulation_state],
        outputs=[evaluation_report]
    )

if __name__ == "__main__":
    demo.launch(css=CUSTOM_CSS)
