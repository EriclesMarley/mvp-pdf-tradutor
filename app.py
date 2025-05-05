
import streamlit as st
import pdfplumber
from transformers import MarianMTModel, MarianTokenizer
from sentence_transformers import SentenceTransformer, util

# Inicializa os modelos uma vez
@st.cache_resource
def load_models():
    pt_en_model_name = 'Helsinki-NLP/opus-mt-pt-en'
    en_pt_model_name = 'Helsinki-NLP/opus-mt-en-pt'

    pt_en_tokenizer = MarianTokenizer.from_pretrained(pt_en_model_name)
    pt_en_model = MarianMTModel.from_pretrained(pt_en_model_name)

    en_pt_tokenizer = MarianTokenizer.from_pretrained(en_pt_model_name)
    en_pt_model = MarianMTModel.from_pretrained(en_pt_model_name)

    sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    return (pt_en_model, pt_en_tokenizer, en_pt_model, en_pt_tokenizer, sbert_model)

pt_en_model, pt_en_tokenizer, en_pt_model, en_pt_tokenizer, sbert_model = load_models()

st.title("Tradução e Análise de PDFs")

uploaded_file = st.file_uploader("Envie um arquivo PDF", type=["pdf"])

if uploaded_file:
    with pdfplumber.open(uploaded_file) as pdf:
        texto = "\n".join(page.extract_text() or '' for page in pdf.pages)

    st.subheader("Texto extraído")
    st.text_area("Texto completo:", texto, height=300)

    option = st.selectbox("Escolha a direção da tradução:", ("Português para Inglês", "Inglês para Português"))

    def traduzir(texto, model, tokenizer):
        tokens = tokenizer.prepare_seq2seq_batch([texto], return_tensors="pt", truncation=True)
        traduzido = model.generate(**tokens)
        return tokenizer.decode(traduzido[0], skip_special_tokens=True)

    if st.button("Traduzir"):
        with st.spinner("Traduzindo..."):
            if option == "Português para Inglês":
                traduzido = traduzir(texto, pt_en_model, pt_en_tokenizer)
            else:
                traduzido = traduzir(texto, en_pt_model, en_pt_tokenizer)
        st.subheader("Tradução")
        st.text_area("Texto traduzido:", traduzido, height=300)

    if st.button("Analisar Similaridade com Resumo"):
        resumo = st.text_area("Cole aqui o resumo para comparação:")
        if resumo:
            emb1 = sbert_model.encode(texto, convert_to_tensor=True)
            emb2 = sbert_model.encode(resumo, convert_to_tensor=True)
            score = util.pytorch_cos_sim(emb1, emb2).item()
            st.success(f"Similaridade: {score:.2f}")
        else:
            st.warning("Por favor, cole um resumo para comparar.")
