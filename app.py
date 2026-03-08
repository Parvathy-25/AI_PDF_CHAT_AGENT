import streamlit as st
import tempfile
import numexpr as ne

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.prompts import PromptTemplate


# ------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------

st.set_page_config(page_title="AI PDF Agent", layout="wide")
st.title("📄 AI PDF Chat Agent")
st.caption("LangChain + RAG + Memory + Calculator Tool + Streamlit UI")
st.write("Upload a PDF and ask questions about it.")


# ------------------------------------------------------------
# Session State
# ------------------------------------------------------------

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = None


# ------------------------------------------------------------
# Calculator Tool
# ------------------------------------------------------------

def calculator_tool(query: str):
    try:
        result = ne.evaluate(query)
        return str(result)
    except Exception:
        return "Calculator error: Invalid expression"

calculator = Tool(
    name="Calculator",
    func=calculator_tool,
    description="Useful for solving mathematical calculations."
)


# ------------------------------------------------------------
# LLM Setup
# ------------------------------------------------------------

try:
    llm = OllamaLLM(model="llama3")
except Exception:
    st.error("⚠️ Ollama is not running. Start Ollama first.")
    st.stop()


# ------------------------------------------------------------
# Prompt Template
# ------------------------------------------------------------

prompt_template = """
You are a PDF document assistant.

Answer the user's question ONLY using the provided context from the uploaded PDF.

Rules:
1. Use ONLY the information in the context.
2. Do NOT use prior knowledge.
3. If the answer is not in the context, say:
   "The answer is not available in the provided PDF."
4. Do not hallucinate or guess.

Context:
{context}

Question:
{question}

Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)


# ------------------------------------------------------------
# PDF Upload
# ------------------------------------------------------------

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file and st.session_state.agent is None:

    try:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Split Text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )

        docs = text_splitter.split_documents(documents)

        # Better Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en"
        )

        # Vector Store
        vectorstore = FAISS.from_documents(docs, embeddings)

        # Retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        

        # Retrieval QA
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        

        # ------------------------------------------------------------
        # PDF Tool
        # ------------------------------------------------------------

        def pdf_qa_tool(query: str):

            try:

                result = qa_chain({"query": query})

                answer = result["result"]

                sources = "\n\n".join(
                    [doc.page_content[:200] for doc in result["source_documents"]]
                )

                return f"{answer}\n\n📄 Source from PDF:\n{sources}"

            except Exception:
                return "Error retrieving answer from PDF."

        pdf_tool = Tool(
            name="PDF_QA",
            func=pdf_qa_tool,
            description="""
            Use this tool to answer ANY question about the uploaded PDF.
            This tool retrieves information directly from the document.
            You must always use this tool for answering PDF questions.
            """
        )

        # ------------------------------------------------------------
        # Agent
        # ------------------------------------------------------------

        agent = initialize_agent(
            tools=[pdf_tool, calculator],
            llm=llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=st.session_state.memory,
            verbose=True
        )

        st.session_state.agent = agent

        st.success("✅ PDF processed successfully. You can now ask questions.")

    except Exception as e:
        st.error(f"⚠️ Error processing PDF: {str(e)}")


# ------------------------------------------------------------
# Display Chat History
# ------------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# ------------------------------------------------------------
# Chat Input
# ------------------------------------------------------------

user_input = st.chat_input("Ask a question about the PDF...")

if user_input:

    if st.session_state.agent is None:
        st.warning("⚠️ Please upload a PDF first.")
        st.stop()

    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    # Generate response
    try:

        response = st.session_state.agent.invoke({"input": user_input})
        answer = response["output"]

    except Exception as e:
        answer = f"⚠️ Error generating response: {str(e)}"

    # Show assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.write(answer)
    # ------------------------------------------------------------
# Expander: Design Choices & Future Extensions
# ------------------------------------------------------------
with st.expander("💡 Design Choices & Future Extensions", expanded=False):
    st.markdown("""
    ### Design Choices
    1. **RAG (Retrieval-Augmented Generation):** PDF is stored in FAISS vector store with embeddings for efficient retrieval.
    2. **Conversation Memory:** Keeps chat context using `ConversationBufferMemory`.
    3. **External Tool Integration:** Calculator tool demonstrates multi-tool capability.
    4. **Prompt Template:** Ensures answers are strictly based on PDF content.
    5. **Error Handling:** Covers PDF processing, LLM initialization, and tool execution.

    ### Future Extensions
    - Add more tools (weather API, web search, summarizer).
    - Support multiple PDFs merged into a single vector store.
    - PDF summarization and table extraction features.
    - Show page numbers in source snippets for better traceability.
    - User authentication and session history storage.
    - Upgrade to a larger LLM for advanced reasoning.
    """)