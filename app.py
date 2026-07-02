import os
import gradio as gr
from dotenv import load_dotenv
from rag_engine import RAGEngine
from simulation import SemhasSimulation, PERSONAS

# Load environment variables
load_dotenv()

# Create custom theme for a premium academic look
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Plus Jakarta Sans"), "system-ui", "sans-serif"]
)

# CSS for a premium, academic-defense look
CUSTOM_CSS = """
.container { max-width: 100%; margin: 0 auto; padding: 15px; }
.header { 
    text-align: center; 
    margin-bottom: 20px; 
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); 
    color: white; 
    padding: 25px 20px; 
    border-radius: 12px; 
    box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    border: 1px solid rgba(255,255,255,0.05);
}
.header h1 { 
    margin: 0; 
    font-size: 2.2em; 
    font-weight: 800; 
    letter-spacing: -0.025em;
    background: linear-gradient(to right, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.header p { 
    margin: 6px 0 0 0; 
    font-size: 1.05em;
    color: #94a3b8; 
}
.sidebar-panel {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 15px;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
}
.examiner-card { 
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); 
    border-left: 5px solid #6366f1; 
    border-radius: 12px; 
    padding: 15px; 
    margin-bottom: 0px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.02);
}
.rag-box { 
    background: #fafaf9; 
    border: 1px solid #e7e5e4; 
    border-left: 5px solid #10b981; 
    border-radius: 12px; 
    padding: 15px;
    height: 480px;
    overflow-y: auto;
    box-shadow: 0 2px 4px rgba(0,0,0,0.02);
}
.feedback-box { 
    background: #faf5ff; 
    border: 1px solid #f3e8ff; 
    border-left: 5px solid #a855f7; 
    border-radius: 12px; 
    padding: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.02);
}
.btn-primary { 
    background: linear-gradient(135deg, #4f46e5 0%, #3730a3 100%) !important; 
    color: white !important; 
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.2) !important;
}
.btn-primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 16px rgba(79, 70, 229, 0.3) !important;
}
.btn-success { 
    background: linear-gradient(135deg, #10b981 0%, #047857 100%) !important; 
    color: white !important; 
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2) !important;
}
.btn-success:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 16px rgba(16, 185, 129, 0.3) !important;
}
#process-status-text {
    margin-top: 10px;
    font-size: 0.9em;
    color: #475569;
}
"""

def process_file_ui(pdf_file, api_key):
    """Callback to process the uploaded PDF file."""
    if not pdf_file:
        return (
            gr.update(value="⚠️ Silakan unggah file PDF thesis Anda."),
            gr.update(value="*Menunggu dokumen untuk preview...*"),
            gr.update(visible=False),
            gr.update(interactive=False),
            None,
            None
        )
    
    # Use API key from UI textbox, fallback to env variable
    active_key = api_key.strip() if api_key else os.getenv("GROQ_API_KEY")
    if not active_key:
        return (
            gr.update(value="❌ Error: API Key tidak ditemukan."),
            gr.update(value="### ❌ Gagal Memproses\nAPI Key tidak ditemukan. Masukkan Groq API Key."),
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
        
        short_status = f"🟢 RAG Sukses! ({chunk_count} Chunks)"
        
        preview_text = f"""### 📄 Abstrak & Deskripsi Dokumen
*   **Nama File**: `{os.path.basename(pdf_file.name)}`
*   **Total Chunks**: `{chunk_count}`
*   **Model Embedding**: `all-MiniLM-L6-v2 (Lokal Offline)`

---
### 🔍 Abstrak Preview:
{abstract_preview}
"""
        
        return (
            gr.update(value=short_status),
            gr.update(value=preview_text),
            gr.update(visible=True),
            gr.update(interactive=True),
            rag_engine,
            simulation
        )
    except Exception as e:
        return (
            gr.update(value=f"❌ Error: {str(e)}"),
            gr.update(value=f"### ❌ Gagal Memproses PDF\n`{str(e)}`"),
            gr.update(visible=False),
            gr.update(interactive=False),
            None,
            None
        )

def start_simulation_ui(simulation, rag):
    """Starts the simulation interview session."""
    if not simulation or not rag:
        return (
            [], 
            "", 
            "", 
            "", 
            "", 
            gr.update(interactive=False), 
            gr.update(interactive=False), 
            gr.update(visible=False),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update()
        )
        
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
        gr.update(visible=True),       # Show active question card
        gr.update(interactive=False), # Disable API key input
        gr.update(interactive=False), # Disable PDF file uploader
        gr.update(interactive=False), # Disable process button
        gr.update(interactive=False), # Disable start button
        gr.update(selected="rag_tab") # Auto-select RAG Context tab!
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
            Uji pemahaman skripsi Anda dengan panel penguji bertenaga AI yang mempelajari dokumen Anda secara mendalam.
            """
        )
        
    with gr.Row(elem_classes="container"):
        
        # Left Column (Setup Steps Wizard)
        with gr.Column(scale=3, elem_classes="sidebar-panel"):
            gr.Markdown("### 📋 Persiapan Sidang")
            
            # Step 1
            api_key_input = gr.Textbox(
                label="1️⃣ Groq API Key (Opsional)", 
                placeholder="Kosongkan jika menggunakan file .env", 
                type="password"
            )
            
            # Step 2
            file_uploader = gr.File(
                label="2️⃣ Unggah PDF Skripsi/Thesis", 
                file_types=[".pdf"],
                height=65
            )
            
            # Step 3 & 4 Actions
            with gr.Row():
                process_btn = gr.Button("3️⃣ Proses RAG", elem_classes="btn-primary")
                start_btn = gr.Button(
                    "4️⃣ Mulai Sidang", 
                    elem_classes="btn-success", 
                    interactive=False
                )
            
            process_status = gr.Markdown("*Status: Menunggu dokumen...*", elem_id="process-status-text")
            
        # Middle Column (Interactive Chat Console)
        with gr.Column(scale=6):
            with gr.Group():
                with gr.Row(visible=False) as session_details_row:
                    current_examiner_card = gr.Markdown(elem_classes="examiner-card")
                    question_status_lbl = gr.Label(label="Status Pertanyaan")
                
                chatbot = gr.Chatbot(
                    label="Ruang Sidang Semhas", 
                    height=500
                )
                
                with gr.Row():
                    student_input = gr.Textbox(
                        show_label=False, 
                        placeholder="Ketik jawaban sanggahan Anda di sini...",
                        interactive=False,
                        scale=9
                    )
                    submit_btn = gr.Button("Kirim", scale=1, elem_classes="btn-primary")
            
        # Right Column (Inspector Panels for Context & Evaluation)
        with gr.Column(scale=3):
            with gr.Tabs() as right_tabs:
                with gr.Tab("📄 Preview Dokumen", id="preview_tab"):
                    document_preview_panel = gr.Markdown(
                        value="*Silakan unggah dan proses skripsi Anda untuk melihat preview dokumen di sini.*",
                        elem_classes="rag-box"
                    )
                    
                with gr.Tab("🔍 RAG Context", id="rag_tab"):
                    rag_context_panel = gr.Markdown(
                        value="*Mulai simulasi untuk melihat context yang dibaca RAG.*", 
                        elem_classes="rag-box"
                    )
                    
                with gr.Tab("📊 Hasil Evaluasi", id="eval_tab"):
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
        outputs=[
            process_status, 
            document_preview_panel, 
            start_btn, 
            start_btn, 
            rag_state, 
            simulation_state
        ]
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
            session_details_row,
            api_key_input,
            file_uploader,
            process_btn,
            start_btn,
            right_tabs
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
    demo.launch(css=CUSTOM_CSS, theme=theme)
