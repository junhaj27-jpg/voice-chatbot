import streamlit as st
import os
from openai import OpenAI
from audio_recorder_streamlit import audio_recorder
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA

# --- [설정] 본인의 API 키를 입력하세요 ---
os.environ["OPENAI_API_KEY"] = "본인의_OpenAI_API_키"
client = OpenAI()
PDF_FILE = "pediatric_emergency.pdf"

st.set_page_config(page_title="소아 응급 보이스 헬퍼", layout="centered")

# RAG 시스템 초기화 (PDF 학습)
@st.cache_resource
def init_rag():
    if not os.path.exists(PDF_FILE):
        st.error(f"'{PDF_FILE}' 파일을 찾을 수 없습니다. 폴더 위치를 확인하세요.")
        return None
    
    loader = PyPDFLoader(PDF_FILE)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(pages)
    
    vectorstore = Chroma.from_documents(
        documents=docs, 
        embedding=OpenAIEmbeddings(),
        collection_name="pediatric_care"
    )
    
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    return RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
    )

qa_bot = init_rag()

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🚑 소아 응급 상황 보이스 챗봇")
st.caption("소아 응급 지침 기반의 전문 답변을 제공합니다.")

# 대화 내용 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 음성 녹음
st.write("---")
audio_bytes = audio_recorder(text="클릭하여 상황 설명하기", icon_size="2x", neutral_color="#ff4b4b")

if audio_bytes:
    with open("temp.mp3", "wb") as f:
        f.write(audio_bytes)
    
    with open("temp.mp3", "rb") as f:
        transcript = client.audio.transcriptions.create(model="whisper-1", file=f)
    
    user_input = transcript.text
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    if qa_bot:
        prompt = f"당신은 소아 응급 전문의입니다. 지침을 바탕으로 답변하세요: {user_input}"
        answer = qa_bot.run(prompt)
    else:
        answer = "PDF 지침서를 로드하지 못했습니다. 일반적인 조언만 가능합니다."

    st.session_state.messages.append({"role": "assistant", "content": answer})
    
    # TTS 생성
    speech = client.audio.speech.create(model="tts-1", voice="nova", input=answer)
    speech.stream_to_file("ans.mp3")
    st.rerun()

if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    st.audio("ans.mp3", format="audio/mp3", autoplay=True)