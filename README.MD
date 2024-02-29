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

ESte projeto utiliza o serviço langsmith para depuração de todo o processo que envolve o engine de IA, análise de custos (US$), erros, chamadas de API, performance, enfim dá pra ver tudo que está acontecendo.

https://smith.langchain.com/


### Roadmap

1- Indexar e exibir as imagens que estiverem em meio aos textos do LinxShare
2- Permitir interações por voz
