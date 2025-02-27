import streamlit as st
import os
import tempfile
import time
import threading
import queue
import concurrent.futures
from openai import OpenAI
from dotenv import load_dotenv
from pydub import AudioSegment
import math

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Configurações do aplicativo
st.set_page_config(
    page_title="Transcritor de Áudio",
    page_icon="🎤",
    layout="wide"
)

# Constantes
MAX_UPLOAD_SIZE_MB = 200
MAX_SEGMENT_SIZE_MB = 25
BYTES_PER_MB = 1024 * 1024
MAX_WORKERS = 3  # Número de workers paralelos para transcrição

# Inicializar estado da sessão se necessário
if 'transcription_progress' not in st.session_state:
    st.session_state.transcription_progress = 0
if 'status_message' not in st.session_state:
    st.session_state.status_message = ""
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Título e descrição
st.title("🎤 Transcritor de Áudio")
st.markdown(f"""
    Faça upload de arquivos de áudio de até {MAX_UPLOAD_SIZE_MB}MB e obtenha sua transcrição completa.
    O arquivo será automaticamente dividido em segmentos menores para processamento paralelo.
""")

# Opções avançadas
with st.expander("Opções avançadas"):
    use_parallel = st.checkbox("Usar processamento paralelo", value=True, 
                             help="Processa múltiplos segmentos simultaneamente para maior velocidade (recomendado)")
    max_workers = st.slider("Número máximo de processamentos paralelos", 1, 5, MAX_WORKERS,
                          help="Mais workers podem acelerar o processamento, mas consomem mais recursos e créditos da API")
    segment_size = st.slider("Tamanho máximo do segmento (MB)", 5, 25, MAX_SEGMENT_SIZE_MB,
                           help="Segmentos menores são processados mais rapidamente, mas podem resultar em mais divisões")

# Obter a chave API do ambiente ou permitir entrada manual
default_api_key = os.getenv("OPENAI_API_KEY", "")

# Se estiver no modo produção e a chave existir no ambiente, use-a diretamente
if default_api_key and os.getenv("STREAMLIT_DEPLOYMENT", "") == "production":
    api_key = default_api_key
    st.success("Chave API configurada via variável de ambiente.")
else:
    # Caso contrário, permita que o usuário insira
    api_key = st.text_input("Insira sua chave API OpenAI", 
                           value=default_api_key,
                           type="password")

# Verificar se a chave API está configurada
if not api_key:
    st.warning("⚠️ Você precisa configurar uma chave API da OpenAI para usar este aplicativo.")
    st.info("💡 A chave API pode ser configurada via variável de ambiente ou inserida no campo acima.")
    st.stop()

# Cache para optimização
@st.cache_data
def get_file_format(file_name):
    """Retorna o formato do arquivo baseado no nome"""
    return file_name.split(".")[-1].lower()

# Função otimizada para dividir o arquivo de áudio em segmentos
def split_audio_file(file_path, segment_size_mb=MAX_SEGMENT_SIZE_MB, status_callback=None):
    """
    Divide um arquivo de áudio em segmentos menores
    
    Args:
        file_path: Caminho para o arquivo de áudio
        segment_size_mb: Tamanho máximo de cada segmento em MB
        status_callback: Função para atualizar o status
        
    Returns:
        Lista de caminhos para os arquivos de segmento
    """
    if status_callback:
        status_callback("Analisando arquivo de áudio...")
    
    # Determinar o formato do arquivo
    file_format = get_file_format(file_path)
    
    # Usar formato otimizado para carregar o áudio
    try:
        # Para formatos específicos, use funções específicas
        if file_format == "mp3":
            audio = AudioSegment.from_mp3(file_path)
        elif file_format == "wav":
            audio = AudioSegment.from_wav(file_path)
        elif file_format == "ogg":
            audio = AudioSegment.from_ogg(file_path)
        else:
            # Para outros formatos, use o método genérico
            audio = AudioSegment.from_file(file_path)
    except Exception as e:
        raise ValueError(f"Erro ao processar arquivo de áudio: {str(e)}")
    
    if status_callback:
        status_callback("Calculando divisão de segmentos...")
    
    # Calcular o número de segmentos necessários
    duration_ms = len(audio)
    file_size = os.path.getsize(file_path)
    
    # Estimar bytes por ms para cálculo de tamanho de segmento
    bytes_per_ms = file_size / duration_ms
    segment_size_ms = int((segment_size_mb * BYTES_PER_MB) / bytes_per_ms)
    
    # Ajustar para garantir que não excedemos o limite
    segment_size_ms = min(segment_size_ms, duration_ms)
    
    # Criar lista para armazenar caminhos dos segmentos
    segment_paths = []
    
    # Divisão mais eficiente em menos segmentos
    segments_count = math.ceil(duration_ms / segment_size_ms)
    
    if status_callback:
        status_callback(f"Dividindo áudio em {segments_count} segmentos...")
    
    # Usar diretório temporário para armazenar segmentos
    temp_dir = os.path.dirname(file_path)
    
    # Dividir o áudio em segmentos de forma otimizada
    for i in range(segments_count):
        start_ms = i * segment_size_ms
        end_ms = min((i + 1) * segment_size_ms, duration_ms)
        
        if status_callback:
            status_callback(f"Criando segmento {i+1} de {segments_count}...")
        
        segment = audio[start_ms:end_ms]
        segment_path = os.path.join(temp_dir, f"segment_{i}.{file_format}")
        
        # Exportar segmento com configurações otimizadas para velocidade
        export_params = {}
        if file_format == "mp3":
            export_params = {"bitrate": "128k"}  # Qualidade mais baixa, mais rápido
        
        segment.export(segment_path, format=file_format, **export_params)
        segment_paths.append(segment_path)
    
    return segment_paths

# Função para transcrever um segmento
def transcribe_segment(segment_path, client, language):
    """
    Transcreve um segmento de áudio usando a API OpenAI
    
    Args:
        segment_path: Caminho para o arquivo de segmento
        client: Cliente OpenAI inicializado
        language: Código do idioma
        
    Returns:
        Texto transcrito e índice do segmento
    """
    segment_index = int(segment_path.split('_')[-1].split('.')[0])
    
    with open(segment_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language=language
        )
    
    return transcript.text, segment_index

# Função para processar transcrição em paralelo
def process_transcription_parallel(segment_paths, client, language, progress_callback=None, status_callback=None):
    """
    Processa transcrições em paralelo
    
    Args:
        segment_paths: Lista de caminhos para segmentos
        client: Cliente OpenAI
        language: Código do idioma
        progress_callback: Função para atualizar progresso
        status_callback: Função para atualizar status
        
    Returns:
        Texto transcrito completo
    """
    total_segments = len(segment_paths)
    results = [None] * total_segments
    completed = 0
    
    # Usar ThreadPoolExecutor para processamento paralelo
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Iniciar todas as tarefas de transcrição
        future_to_segment = {
            executor.submit(transcribe_segment, segment_path, client, language): segment_path
            for segment_path in segment_paths
        }
        
        # Processar os resultados conforme concluídos
        for future in concurrent.futures.as_completed(future_to_segment):
            segment_path = future_to_segment[future]
            try:
                transcript_text, segment_index = future.result()
                results[segment_index] = transcript_text
                
                # Limpar arquivo após uso
                try:
                    os.remove(segment_path)
                except:
                    pass
                
                # Atualizar progresso
                completed += 1
                if progress_callback:
                    progress_percent = 0.1 + (completed / total_segments) * 0.8
                    progress_callback(progress_percent)
                
                if status_callback:
                    status_callback(f"Transcrito {completed}/{total_segments} segmentos")
                
            except Exception as e:
                if status_callback:
                    status_callback(f"Erro no segmento {segment_path}: {str(e)}")
    
    # Juntar resultados na ordem correta
    full_transcript = " ".join([r for r in results if r is not None])
    return full_transcript

# Função para processar transcrição em sequência
def process_transcription_sequential(segment_paths, client, language, progress_callback=None, status_callback=None):
    """
    Processa transcrições sequencialmente
    
    Args:
        segment_paths: Lista de caminhos para segmentos
        client: Cliente OpenAI
        language: Código do idioma
        progress_callback: Função para atualizar progresso
        status_callback: Função para atualizar status
        
    Returns:
        Texto transcrito completo
    """
    total_segments = len(segment_paths)
    full_transcript = ""
    
    for i, segment_path in enumerate(segment_paths):
        if status_callback:
            status_callback(f"Transcrevendo segmento {i+1} de {total_segments}...")
        
        transcript_text, _ = transcribe_segment(segment_path, client, language)
        full_transcript += transcript_text + " "
        
        # Limpar arquivo após uso
        try:
            os.remove(segment_path)
        except:
            pass
        
        # Atualizar progresso
        if progress_callback:
            progress_percent = 0.1 + (i + 1) / total_segments * 0.8
            progress_callback(progress_percent)
    
    return full_transcript.strip()

# Função para lidar com o progresso
def update_progress(value):
    st.session_state.transcription_progress = value

# Função para atualizar mensagem de status
def update_status(message):
    st.session_state.status_message = message

# Upload de arquivo
uploaded_file = st.file_uploader(
    f"Escolha um arquivo de áudio (até {MAX_UPLOAD_SIZE_MB}MB)", 
    type=["mp3", "wav", "m4a", "ogg", "flac"]
)

# Opções para idioma de transcrição
idioma = st.selectbox(
    "Selecione o idioma da transcrição",
    options=["pt", "en", "es", "fr", "de", "it", "ja", "ko", "zh"],
    index=0
)

# Mostrar progresso se estiver processando
if st.session_state.processing:
    # Exibir barra de progresso
    st.progress(st.session_state.transcription_progress)
    st.info(st.session_state.status_message)

# Quando o usuário clicar no botão de transcrição
if st.button("Transcrever", disabled=st.session_state.processing) and uploaded_file is not None:
    # Verificar o tamanho do arquivo
    file_size_mb = uploaded_file.size / BYTES_PER_MB
    
    if file_size_mb > MAX_UPLOAD_SIZE_MB:
        st.error(f"O arquivo é muito grande! O tamanho máximo permitido é {MAX_UPLOAD_SIZE_MB}MB.")
        st.stop()
    
    # Marcar como processando
    st.session_state.processing = True
    st.session_state.transcription_progress = 0
    st.session_state.status_message = "Iniciando processamento..."
    
    # Atualizar UI imediatamente
    st.rerun()

# Função principal de processamento (será executada após o rerun quando processing=True)
if st.session_state.processing and uploaded_file is not None:
    try:
        # Configurar cliente OpenAI
        client = OpenAI(api_key=api_key)
        
        # Criar diretório temporário para os arquivos
        with tempfile.TemporaryDirectory() as temp_dir:
            # Salvar o arquivo temporariamente
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            update_status("Analisando arquivo de áudio...")
            
            # Verificar tamanho do arquivo
            file_size_mb = os.path.getsize(temp_file_path) / BYTES_PER_MB
            
            # Se o arquivo for menor que o limite da API, processar diretamente
            if file_size_mb <= segment_size:
                update_status("Transcrevendo arquivo (único segmento)...")
                update_progress(0.2)
                
                with open(temp_file_path, "rb") as audio_file:
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language=idioma
                    )
                
                full_transcript = transcript.text
                update_progress(1.0)
            else:
                # Dividir o arquivo em segmentos
                update_status("Dividindo arquivo em segmentos...")
                update_progress(0.1)
                
                segment_paths = split_audio_file(
                    temp_file_path, 
                    segment_size_mb=segment_size,
                    status_callback=update_status
                )
                
                total_segments = len(segment_paths)
                update_status(f"Arquivo dividido em {total_segments} segmentos. Iniciando transcrição...")
                
                # Escolher método de processamento
                if use_parallel and total_segments > 1:
                    full_transcript = process_transcription_parallel(
                        segment_paths, 
                        client, 
                        idioma,
                        progress_callback=update_progress,
                        status_callback=update_status
                    )
                else:
                    full_transcript = process_transcription_sequential(
                        segment_paths, 
                        client, 
                        idioma,
                        progress_callback=update_progress,
                        status_callback=update_status
                    )
                
                update_progress(0.9)
                update_status("Finalizando transcrição...")
                time.sleep(0.5)  # Pequena pausa para UX
            
            # Atualizar progresso
            update_progress(1.0)
            update_status("Transcrição concluída!")
            
            # Exibir resultado
            st.success("Transcrição concluída com sucesso!")
            st.subheader("Resultado da transcrição:")
            st.text_area("Texto transcrito", full_transcript, height=300)
            
            # Opção para baixar a transcrição
            st.download_button(
                label="Baixar transcrição como arquivo TXT",
                data=full_transcript,
                file_name=f"{os.path.splitext(uploaded_file.name)[0]}_transcricao.txt",
                mime="text/plain"
            )
            
            # Resetar estado de processamento
            st.session_state.processing = False
            
    except Exception as e:
        st.error(f"Ocorreu um erro durante a transcrição: {str(e)}")
        if "Invalid file format." in str(e):
            st.warning("O formato do arquivo pode não ser suportado pela API do Whisper ou estar corrompido.")
        elif "maximum allowed size" in str(e):
            st.warning("Mesmo após a divisão, um dos segmentos pode estar muito grande. Tente um arquivo menor ou usar segmentos menores.")
        
        # Resetar estado de processamento em caso de erro
        st.session_state.processing = False

# Adicionar instruções e informações adicionais
with st.expander("Como funciona o processamento de arquivos grandes?"):
    st.markdown("""
    ### Processo de divisão e transcrição:

    1. **Upload**: Você faz upload de um arquivo de áudio de até 200MB
    2. **Análise**: O sistema verifica o tamanho do arquivo
    3. **Divisão**: Se necessário, o arquivo é dividido em segmentos menores
    4. **Processamento paralelo**: Vários segmentos são transcritos simultaneamente
    5. **Combinação**: As transcrições de todos os segmentos são combinadas
    6. **Resultado**: Você recebe a transcrição completa para download

    ### Otimização de performance:

    - **Processamento paralelo**: Transcreve múltiplos segmentos simultaneamente
    - **Tamanho de segmento ajustável**: Permite balancear velocidade e precisão
    - **Exportação otimizada**: Usa configurações de compressão eficientes
    """)

# Instruções e informações adicionais
st.markdown("---")
st.markdown("""
### Como usar:
1. Faça upload de um arquivo de áudio (até 200MB)
2. Selecione o idioma do áudio
3. Ajuste as opções avançadas se necessário
4. Clique em "Transcrever"
5. Acompanhe o progresso da transcrição
6. Baixe o resultado como arquivo TXT

### Formatos suportados:
- MP3, WAV, M4A, OGG, FLAC
""")

# Rodapé
st.markdown("---")
st.markdown("Desenvolvido com Streamlit e OpenAI Whisper API | Versão otimizada para performance")
