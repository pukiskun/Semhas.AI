---
title: Semhas.AI
emoji: 🎓
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.19.0
app_file: app.py
pinned: false
---

# 🎓 Semhas.AI - RAG-Powered Thesis Defense Simulator

Semhas.AI adalah aplikasi simulator sidang hasil (Semhas) atau sidang skripsi bertenaga AI. Aplikasi ini menggunakan teknologi **RAG (Retrieval-Augmented Generation)** lokal untuk membaca dokumen PDF thesis/skripsi pengguna secara mendalam dan mensimulasikan panel penguji akademis yang kritis secara interaktif.

Proyek ini dirancang hemat biaya (**$0 server cost**) dengan menjalankan proses embedding/RAG secara lokal (offline) dan menggunakan model **Qwen 3.6 27B** di Groq API untuk dialog interaktif.

---

## 🚀 Fitur Utama

- **Local Offline RAG**: Pemrosesan dokumen PDF, chunking, dan pembuatan vector database dilakukan 100% lokal di komputer Anda menggunakan model `all-MiniLM-L6-v2` dari Hugging Face (Bebas biaya cloud & kuota API).
- **Multi-Persona Examiner Panel**: Simulasi diuji oleh 3 karakter dosen penguji dengan sifat unik:
  1. **Prof. Ahmad Yani (Subject Matter Expert)**: Menguji teori dasar dan kebaruan penelitian.
  2. **Dr. Sarah Fitri (Methodologist)**: Menguji dataset, parameter model, statistik, dan metodologi.
  3. **Dr. Edward Hutapea (The Skeptic)**: Menguji limitasi, kegagalan sistem, dan kontribusi nyata penelitian.
- **RAG Sandbox Preview**: Sidebar visual yang memperlihatkan potongan paragraf skripsi mana yang sedang dibaca oleh AI saat merumuskan pertanyaan.
- **Grading & Evaluation Checklist**: Laporan penilaian komprehensif (Skala IPK A-E) beserta daftar revisi skripsi di akhir sesi.

---

## 🛠️ Cara Instalasi & Menjalankan

Bagi siapa saja yang meng-clone project ini, berikut langkah-langkah untuk menjalankannya:

### 1. Prasyarat (Prerequisites)
Pastikan Anda sudah menginstal Python (versi 3.8 - 3.11 direkomendasikan) pada komputer Anda.

### 2. Clone Repositori
```bash
git clone https://github.com/pukiskun/Semhas.AI.git
cd Semhas.AI
```

### 3. Instal Dependencies
Jalankan perintah berikut di terminal untuk menginstal pustaka yang diperlukan:
```bash
pip install -r requirements.txt
```
*(Catatan: Penginstalan pertama kali mungkin memakan waktu beberapa menit karena akan mengunduh package PyTorch).*

### 4. Konfigurasi API Key
1. Dapatkan API Key gratis di **[Groq Console](https://console.groq.com/)**.
2. Duplikat file `.env.example` menjadi `.env`:
   ```bash
   cp .env.example .env
   ```
3. Buka file `.env` dan masukkan API Key Anda:
   ```env
   GROQ_API_KEY=gsk_your_actual_key_here
   ```

### 5. Jalankan Aplikasi
Jalankan aplikasi Gradio dengan perintah:
```bash
python app.py
```
Buka browser Anda dan akses tautan lokal yang tertera di terminal (biasanya **[http://127.0.0.1:7860](http://127.0.0.1:7860)**).

---

## 💡 Tech Stack
- **Frontend & App Interface**: Gradio
- **Embedding & Vector Search**: Hugging Face `sentence-transformers` (`all-MiniLM-L6-v2` model) & NumPy (untuk cosine similarity lokal)
- **Large Language Model (LLM)**: `qwen/qwen3.6-27b` via Groq SDK
- **PDF Parser**: PyPDF
