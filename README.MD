### Este projeto pode ser executado de duas formas:

chat_console.py: Permite testar o chat via console e recriar a base de conhecimento

chat_web.py: executa a interface do chat ChainLit


### Base de Conhecimento

A base de dados de postos já está carregada no bando vetorial do Tito. É preciso mais tarde mover para um outro banco de dados vetorial, de propriedade da Linx, provavelmente no Azure.

Para carregar outras bases basta alterar os arquivos info_confluence.txt e info_youtube.txt e executar a carga da base de dados novamente, através da opção correspondente em chat_console.py


### Para instalar o chat dentro de uma janela (webchat)

https://docs.chainlit.io/deployment/copilot

This example assumes your Chainlit server is running on http://localhost:8000

<head>
  <meta charset="utf-8" />
</head>
<body>
  <!-- ... -->
  <script src="http://localhost:8000/copilot/index.js"></script>
  <script>
    window.mountChainlitWidget({
      chainlitServer: "http://localhost:8000",
    });
  </script>
</body>


### GetLiteral

Este projeto utiliza o serviço getliteral que armazena automaticamente todas as conversas que acontecerem neste chat para análise posterior.

https://cloud.getliteral.ai/projects/Linx%20Assistente%20IA-X1JYIHhvJDed/settings


### LangSmith

ESte projeto utiliza o serviço langsmith para depuração de todo o processo que envolve o engine de IA, análise de custos (US$), erros, chamadas de API, performance, enfim dá pra ver tudo que está acontecendo. O LangSmith é uma ferramenta do stack LangChain.

https://smith.langchain.com/


### Roadmap

1- Indexar e exibir as imagens que estiverem em meio aos textos do LinxShare
2- Permitir interações por voz
3- Resolver bug que está impedindo o streaming das respostas


### Significado das principais chaves do arquivo .env

# Chaves para configurar projeto no LangSmith
LANGCHAIN_API_KEY="ls__dbd18ca08696479aa04dccd04740e792"
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_PROJECT="Linx Assistente IA"  

# Chave para configurar projeto no GetLiteral
LITERAL_API_KEY="cl_XgSPdyEvRcp9ssOTzawR+jyEQ/yoeoTJYe3nkJPylHA="

LINX_PRODUTO="Linx Seller"
OPENAI_API_KEY=""
MODEL_NAME="gpt-3.5-turbo"
USE_AZURE= "N"
AZURE_OPENAI_API_TYPE= "azure"
AZURE_OPENAI_API_VERSION= "2023-05-15"
AZURE_OPENAI_API_BASE= "https://your-resource-name.openai.azure.com"
AZURE_OPENAI_API_KEY= "<your Azure OpenAI API key>"
AZURE_DEPLOYMENT_NAME= "gpt-35-turbo" 

# Chaves para configurar gravação no banco de dados vetorial. 
# Elas precisarão ser substituídas pelo banco de que a Linx escolher.
# Neste momento será necessário criar uma nova rotina como a save_vectorstore_qdrant
QDRANT_URL=""
QDRANT_API_KEY=""

# COnfigurações do banco vetorial
CHUNK_SIZE=750
CHUNK_OVERLAP=150
NOME_BASE_VETORIAL_V1="BASE_LINX_SELLER_V1"
NOME_BASE_VETORIAL_V2="BASE_LINX_SELLER_V2"
NOME_BASE_VETORIAL_ATIVA ="BASE_LINX_SELLER_V1"
CRIACAO_BASE_VETORIAL_PERMITIDA="N"

# Configurações para acesso ao LinxShare por API. Necessário solicitar para a Ellen o acesso.
CONFLUENCE_URL="https://share.linx.com.br/"
CONFLUENCE_USERNAME=""
CONFLUENCE_PASSWORD=""
CONFLUENCE_NIVEL_RECURSIVIDADE=5

# Configurações de pesquisa ao banco de dados vetorial
SEARCH_K=5
SEARCH_MIN_SCORE=0.8

# Configuração da memória do chat
DB_CACHE_URL="sqlite:///chat_history.db"

# Configuração do usuário autenticado.
# O pograma que executa este chat deverá cuidar da autenticação do usuário e passar o email do usuário autenticado para o chat.
EMAIL_USUARIO="jose@linx.com.br"

# Indica se o assistente poderá usar os seus conhecimentos ou apenas o contexto
CHAT_RESTRITO_AO_CONTEXTO="S"

VERBOSE="S"

# Configuração para manipulação das playlists e canais do Youtube
GOOGLE_API_KEY=""

# Chaves necessários para execução de web scraping. Não utilizado por enquanto.
SCRAPE_API_KEY=""
SCRAPE_PROXY=""