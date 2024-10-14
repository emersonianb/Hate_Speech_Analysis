# Hate Speech Analysis in Tweets

📝 Descrição

A Análise de Discurso de Ódio em Tweets é um pequeno exercício que visa identificar e classificar automaticamente discursos de ódio em publicações no Twitter. Utilizando dados reais obtidos do site data.world, este estudo aplica redes neurais sequenciais para realizar a classificação entre três categorias:

    🟥 Discurso de Ódio (Hate Speech)
    🟧 Linguagem Ofensiva
    🟩 Linguagem Neutra

Este projeto faz parte dos estudos na disciplina TIA - Tópicos em Inteligência Artificial na Universidade de Coimbra.

🚀 Tecnologias Utilizadas

    Python 3.8+
    Keras e TensorFlow para construção e treino de redes neurais
    pandas para manipulação de dados
    TfidfVectorizer para a vetorização dos tweets
    Jupyter Notebook para análise e experimentação

📊 Modelo de Classificação

O modelo utilizado é uma rede neural sequencial com várias camadas densas e dropout, focada em realizar a classificação dos tweets em três classes. A arquitetura inclui:

    Camada de entrada: Vetorização dos tweets utilizando TF-IDF.
    Camadas ocultas: Camadas densas com unidades de ativação ReLU.
    Camada de saída: Classificação com 3 neurônios e função de ativação Softmax para as 3 classes (ódio, ofensiva, neutra).

📈 Resultados Obtidos

A performance do modelo é avaliada utilizando métricas como:

    Acurácia
    Precisão

Os resultados são visualizados utilizando gráficos de barras para melhor entendimento das métricas por classe.

🧠 Como Usar

    O projeto ainda não está livre para uso, porque ainda se encontra em desenvolvimento para melhoramento, mas em breve será disponibilizado! :)

👨‍💻 Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues e pull requests.

📫 Contato

    Autor: Emerson Sousa
    Email: emersonianbezerra@gmail.com
    LinkedIn: https://www.linkedin.com/in/emerson-sousa-054a951b0
