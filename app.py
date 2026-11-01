import streamlit as st
import pandas as pd
from src.predictor import HateSpeechPredictor

st.set_page_config(
    page_title="Análise de Discurso de Ódio em Tweets",
    page_icon="🛡️",
    layout="centered"
)

@st.cache_resource
def load_predictor():
    return HateSpeechPredictor()

try:
    predictor = load_predictor()
except Exception as e:
    st.error(f"Erro ao carregar o modelo: {e}")
    st.info("Execute 'python export_model.py' para gerar os arquivos na pasta 'models/'.")
    st.stop()

st.title("🛡️ Análise de Discurso de Ódio em Tweets")
st.caption("Contexto Acadêmico — Tópicos em Inteligência Artificial (TIA) | Universidade de Coimbra")

# AVISO EM DESTAQUE SOBRE O IDIOMA
st.warning("🌐 **Aviso de Idioma:** Este modelo foi treinado com um dataset em **inglês**. Para obter resultados precisos, insira os tweets em inglês.")

st.markdown("""
Esta aplicação classifica o texto de tweets em três categorias:
* 🚨 **Discurso de Ódio**: Expressões de ódio dirigidas a grupos protegidos.
* ⚠️ **Linguagem Ofensiva**: Uso de palavrões ou linguagem vulgar sem incitação direta de ódio.
* ✅ **Linguagem Neutra**: Texto sem conteúdo ofensivo ou de ódio.
""")

st.divider()

option = st.radio("Selecione o Modo de Análise:", ["Tweet Único", "Análise em Lote"], horizontal=True)

if option == "Tweet Único":
    tweet_input = st.text_area(
        "Digite o texto do tweet (em inglês):", 
        placeholder="Exemplo: I really enjoyed the football match yesterday...", 
        height=120
    )
    
    if st.button("Analisar Tweet", type="primary"):
        if not tweet_input.strip():
            st.warning("Por favor, digite um texto válido.")
        else:
            with st.spinner("Classificando texto..."):
                res = predictor.predict(tweet_input)
                
            label = res['label']
            scores = res['scores']
            
            st.subheader("Resultado da Classificação")
            if label == "Discurso de Ódio":
                st.error(f"🚨 **Categoria Predominante:** {label}")
            elif label == "Linguagem Ofensiva":
                st.warning(f"⚠️ **Categoria Predominante:** {label}")
            else:
                st.success(f"✅ **Categoria Predominante:** {label}")
                
            st.markdown("### Distribuição de Pontuação do Modelo")
            st.caption("Nota: As pontuações representam as saídas brutas de probabilidade do modelo.")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Discurso de Ódio", f"{scores['Discurso de Ódio']*100:.1f}%")
            col2.metric("Linguagem Ofensiva", f"{scores['Linguagem Ofensiva']*100:.1f}%")
            col3.metric("Linguagem Neutra", f"{scores['Linguagem Neutra']*100:.1f}%")
            
            st.write("**Detalhamento Visual:**")
            for cat, prob in scores.items():
                st.text(f"{cat}: {prob*100:.2f}%")
                st.progress(min(max(prob, 0.0), 1.0))

else:
    batch_input = st.text_area(
        "Digite múltiplos tweets em inglês (um por linha):",
        placeholder="Tweet 1 in English\nTweet 2 in English\nTweet 3 in English",
        height=180
    )
    
    if st.button("Analisar Lote", type="primary"):
        lines = [line.strip() for line in batch_input.split("\n") if line.strip()]
        if not lines:
            st.warning("Insira pelo menos uma linha de texto válida.")
        else:
            with st.spinner("Processando lote..."):
                results = predictor.predict(lines)
                
            data = []
            for r in results:
                data.append({
                    "Tweet": r['tweet'],
                    "Categoria Predita": r['label'],
                    "Score Ódio": f"{r['scores']['Discurso de Ódio']:.3f}",
                    "Score Ofensivo": f"{r['scores']['Linguagem Ofensiva']:.3f}",
                    "Score Neutro": f"{r['scores']['Linguagem Neutra']:.3f}"
                })
            
            df_res = pd.DataFrame(data)
            st.dataframe(df_res, use_container_width=True)

st.divider()

with st.expander("⚠️ Limitações do Modelo e Considerações Éticas"):
    st.write("""
    - **Idioma Suportado:** O modelo foi projetado e avaliado exclusivamente para o idioma **inglês**.
    - **Limitações de Contexto:** Classificadores automáticos têm dificuldade em identificar sarcasmo, ironia e gírias locais.
    - **Falsos Positivos:** O uso de termos vulgares pode ser classificado como linguagem ofensiva ou discurso de ódio sem considerar o contexto interpessoal.
    - **Necessidade de Revisão Humana:** Este modelo é um recurso demonstrativo e não deve ser utilizado como único critério para moderação automática de conteúdo.
    """)