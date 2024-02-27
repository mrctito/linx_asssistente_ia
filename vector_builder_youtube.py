import os
from typing import List, Tuple
import PyPDF2
from io import BytesIO
import json
from langchain.agents import AgentType
from langchain_community.llms import OpenAI
from langchain.agents import initialize_agent
from langchain.agents.agent_toolkits.jira.toolkit import JiraToolkit
from langchain.schema import Document
from langchain.text_splitter import (CharacterTextSplitter,
                                    RecursiveCharacterTextSplitter)
from atlassian import Confluence
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.document_transformers.openai_functions import create_metadata_tagger
from langchain_community.vectorstores.qdrant import Qdrant
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from langchain.document_loaders.sitemap import SitemapLoader
from qdrant_client import QdrantClient, models
from langchain.indexes import SQLRecordManager, index
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build

from lib_utils import carregar_arquivo_para_dicionario


def get_playlist_videos(playlist_id, api_key):

    youtube = build('youtube', 'v3', developerKey=api_key)
    
    videos = []
    next_page_token = None

    while True:
        response = youtube.playlistItems().list(
            playlistId=playlist_id,
            part='snippet',
            maxResults=50,  # API allows max 50 items per request
            pageToken=next_page_token
        ).execute()

        for item in response['items']:
            video_id = item['snippet']['resourceId']['videoId']
            videos.append(video_id)
        
        next_page_token = response.get('nextPageToken')

        if not next_page_token:
            break

    return videos


def get_channel_videos(channel_id, api_key):

    youtube = build('youtube', 'v3', developerKey=api_key)

    response = youtube.channels().list(
        id=channel_id,
        part='contentDetails'
    ).execute()

    uploads_playlist_id = response['items'][0]['contentDetails']['relatedPlaylists']['uploads']

    # Agora, vamos buscar os vídeos dessa playlist
    playlist_response = youtube.playlistItems().list(
        playlistId=uploads_playlist_id,
        part='contentDetails',
        maxResults=50  # Você pode ajustar o número de resultados desejados
    ).execute()

    videos = [item['contentDetails']['videoId'] for item in playlist_response['items']]
 
    return videos


def extract_video_code(url):
    parsed_url = urlparse(url)
    query_params = parse_qsl(parsed_url.query)
    for item in query_params:
        if item[0] == 'v':
            id = item[1]
            return id
    return None


def load_youtube_docs(video_id):
    if video_id:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['pt', 'en', 'fr', 'es'])
        url = f'https://www.youtube.com/watch?v={video_id}'
        metadata = {
            "title": 'Youtube',
            "source": url,
        }
        full_transcript = ' '.join(item['text'] for item in transcript)
        result = Document(page_content=full_transcript, metadata=metadata)
        return result
    return None


def processa_videos(dados: dict) -> List[Document]:
    google_api_key = os.getenv("GOOGLE_API_KEY")
    documents = []
    lista_videos = set()
    palavras_total = 0        
    total_bytes = 0
    videos_total = 0

    channels = dados.get('channel')
    if channels is not None:    
        print("Processando channels")
        for channel in channels:
            channel_videos = get_channel_videos(channel, google_api_key)
            for video_id in channel_videos:
                lista_videos.add(video_id) 

    playlist = dados.get('playlist')
    if playlist is not None:    
        print("Processando playlists")
        for channel in playlist:
            channel_videos = get_playlist_videos(channel, google_api_key)
            for video_id in channel_videos:
                lista_videos.add(video_id) 

    video_urls = dados.get('url')
    if video_urls is not None:
        print("Processando video urls")
        for url in video_urls:
            video_id = extract_video_code(url)
            lista_videos.add(video_id)

    for video_id in lista_videos:
        try:
            doc = load_youtube_docs(video_id)
            if doc:
                documents.append(doc)
                palavras = doc.page_content.split()
                palavras_total = palavras_total + len(palavras)
                videos_total += 1
                print(video_id, "OK", f"Video youtube processado - total {len(palavras)} palavras.")
            else:
                print(video_id, "ERRO", f"Video youtube NÃO processado - sem conteúdo.")
        except Exception as e:
            print(video_id, "ERRO", f"Video youtube NÃO processado - Erro:{str(e)}")

    print(f"Processadas: {videos_total} videos - total {palavras_total} palavras")
    return documents


async def processa_dados_youtube():
    chunk_size = int(os.getenv("CHUNK_SIZE"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP"))

    caminho_arquivo = "info_youtube.txt"
    dados = carregar_arquivo_para_dicionario(caminho_arquivo)
    documents = processa_videos(dados)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                chunk_overlap=chunk_overlap,
                                                separators= ["\n\n", "\n", ".", ";", ",", " ", ""],
                                                length_function=len)
    
    chunks = text_splitter.split_documents(documents)
    print(f"Youtube: total {len(chunks)} chunks")
    return chunks