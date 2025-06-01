import streamlit as st

# 1. set_page_config HARUS setelah import streamlit tapi sebelum import lainnya
st.set_page_config(
    page_title="Your App Title",
    page_icon="🧊",
    layout="wide"
)

# 2. Kode untuk menghilangkan menu
hide_menu = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_menu, unsafe_allow_html=True)

# 3. Mengatasi konflik PyTorch dengan Streamlit
import os
os.environ["STREAMLIT_SERVER_WATCH_CHANGES"] = "false"

# 4. import dependencies lainnya
import tempfile
import logging
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Konfigurasi logging
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Custom CSS untuk tampilan yang lebih baik
st.markdown("""
    <style>
    .stApp {
    max-width: 1200px;
    margin: 0 auto;
    }
    .chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    }
    .user-message {
    background-color: #f0f2f6;
    }
    .assistant-message {
    background-color: #e8f0fe;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_embeddings():
    """Load embeddings model with caching"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}
    )

def initialize_llm():
    """Inisialisasi model LLM dengan konfigurasi yang dioptimalkan"""
    try:
        model_config = {
            "temperature": 0.27,  # Lebih rendah untuk jawaban yang lebih konsisten
            "max_tokens": 3600,  # Ditingkatkan untuk jawaban yang lebih lengkap
            "top_p": 0.9,  # Parameter tambahan untuk kualitas output
            "presence_penalty": 0.1,  # Mendorong variasi dalam respons
            "frequency_penalty": 0.1  # Mengurangi pengulangan
        }

        llm = ChatGroq(
            api_key=st.secrets["GROQ_API_KEY"],
            model_name="gemma2-9b-it",
            **model_config
        )
        return llm
    except Exception as e:
        logging.error(f"Error initializing LLM: {e}")
        raise

def process_documents(pdf_docs):
    """Memproses dokumen PDF dengan penanganan error yang lebih baik"""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=450,  # Ukuran chunk yang dioptimalkan
            chunk_overlap=150,  # Overlap yang lebih besar untuk konteks yang lebih baik
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]  # Separator yang lebih detail
        )

        documents = []
        temp_files = []

        for pdf in pdf_docs:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                    temp_file.write(pdf.read())
                    temp_files.append(temp_file.name)
                    loader = PyPDFLoader(temp_file.name)
                    documents.extend(loader.load())
            except Exception as pdf_error:
                logging.error(f"Error processing PDF {pdf.name}: {pdf_error}")
                continue

        # Cleanup temp files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass

        if not documents:
            raise ValueError("Tidak ada dokumen yang berhasil diproses")

        chunks = text_splitter.split_documents(documents)

        # Menggunakan model embedding multibahasa yang lebih baik
        embeddings = load_embeddings()

        vector_store = FAISS.from_documents(chunks, embeddings)
        return vector_store

    except Exception as e:
        logging.error(f"Error in document processing: {e}")
        raise

def get_conversation_chain(vector_store):
    """Membuat rantai percakapan dengan konfigurasi yang dioptimalkan"""
    try:
        llm = initialize_llm()

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )

        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(
                search_kwargs={"k": 3}  # Mengambil 3 dokumen teratas
            ),
            memory=memory,
            return_source_documents=True,
            verbose=False  # Ubah ke True hanya untuk debugging
        )

        return conversation_chain

    except Exception as e:
        logging.error(f"Error creating conversation chain: {e}")
        raise

def main():
    st.title("🧠 PRAJNA DocQA")
    st.markdown(
        """
        <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem;'>
        <p style='font-size: 18px; margin: 0;'>
        <strong>Selamat datang di PRAJNA DocQA!</strong> - Asisten Pribadi Anda untuk analisis dokumen PDF.
        Unggah dokumen Anda dan ajukan pertanyaan untuk mendapatkan wawasan yang mendalam. Pastikan Anda memiliki koneksi internet yang baik.
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Sidebar dengan tampilan yang lebih baik
    with st.sidebar:
        st.header("📁 Unggah Dokumen")
        pdf_docs = st.file_uploader(
            "Pilih file PDF Anda",
            type=["pdf"],
            accept_multiple_files=True,
            help="Anda dapat mengunggah beberapa file PDF sekaligus"
        )

        if st.button("🔄 Proses Dokumen", use_container_width=True):
            if not pdf_docs:
                st.error("Silakan unggah dokumen terlebih dahulu!")
                return

            with st.spinner("📊 Memproses dokumen..."):
                try:
                    st.session_state.vector_store = process_documents(pdf_docs)
                    st.session_state.conversation = get_conversation_chain(
                        st.session_state.vector_store
                    )
                    st.success("✅ Dokumen berhasil diproses!")
                except Exception as e:
                    st.error(f"❌ Terjadi kesalahan: {str(e)}")
                    logging.error(f"Error in main processing: {e}")

    # Inisialisasi riwayat chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Tampilan chat yang lebih baik
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input chat dengan penanganan error yang lebih baik
    if "conversation" not in st.session_state:
        st.info("ℹ️ Silakan unggah dan proses dokumen PDF untuk memulai percakapan.")
    else:
        if prompt := st.chat_input("💭 Ketik pertanyaan Anda di sini..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                try:
                    modified_prompt = (
                        "Berikan jawaban selalu dalam bahasa Indonesia yang jelas dan terstruktur, "
                        "kecuali jika diminta dalam bahasa Inggris. Jawaban harus mencakup: "
                        f"{prompt}"
                    )

                    with st.spinner("🤔 Sedang berpikir..."):
                        response = st.session_state.conversation.invoke({"question": modified_prompt})
                        st.markdown(response["answer"])
                        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

                except Exception as e:
                    error_msg = f"❌ Maaf, terjadi kesalahan dalam memproses pertanyaan Anda: {str(e)}"
                    st.error(error_msg)
                    logging.error(f"Error in chat response: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # Footer dengan disclaimer yang lebih menarik
    st.markdown(
        """
        <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; margin-top: 2rem;'>
        <p style='font-size: 12px; color: #6c757d; margin: 0;'>
        ⚠️ <strong>Disclaimer:</strong> AI-LLM dapat membuat kesalahan.
        Mohon verifikasi informasi penting sebelum mengambil keputusan.
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
