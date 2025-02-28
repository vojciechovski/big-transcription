import streamlit as st
import os
import tempfile
import time
import subprocess
import shutil
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
MAX_SEGMENT_SIZE_MB = 20  # Reduzido de 25MB para 20MB para ter margem de segurança
BYTES_PER_MB = 1024 * 1024
MAX_API_SIZE_BYTES = 24 * 1024 * 1024  # 24MB como limite real da API (para ter margem)

# Título e descrição
st.title("🎤 Transcritor de Áudio")
st.markdown(f"""
    Faça upload de arquivos de áudio de até {MAX_UPLOAD_SIZE_MB}MB e obtenha sua transcrição completa.
    O arquivo será automaticamente convertido para um formato compatível e dividido em segmentos menores para processamento.
""")

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

# Upload de arquivo
uploaded_file = st.file_uploader(
    f"Escolha um arquivo de áudio (até {MAX_UPLOAD_SIZE_MB}MB)", 
    type=["mp3", "wav", "m4a", "ogg", "flac", "aac"]
)

# Opções para idioma de transcrição
idioma = st.selectbox(
    "Selecione o idioma da transcrição",
    options=["pt", "en", "es", "fr", "de", "it", "ja", "ko", "zh"],
    index=0
)

# Funções auxiliares

def check_ffmpeg_installed():
    """Verifica se o ffmpeg está instalado e funcional"""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        return result.returncode == 0
    except:
        return False

def convert_audio_to_wav(input_path, output_dir=None):
    """
    Converte um arquivo de áudio para WAV usando ffmpeg diretamente
    
    Args:
        input_path: Caminho para o arquivo de entrada
        output_dir: Diretório para salvar o arquivo convertido (opcional)
        
    Returns:
        Caminho para o arquivo WAV convertido
    """
    # Determinar o diretório de saída
    if output_dir is None:
        output_dir = os.path.dirname(input_path)
    
    # Gerar o nome do arquivo de saída
    output_filename = os.path.splitext(os.path.basename(input_path))[0] + ".wav"
    output_path = os.path.join(output_dir, output_filename)
    
    # Comando para converter para WAV
    try:
        # Tentar usando processo direto do ffmpeg (mais confiável)
        cmd = [
            "ffmpeg", 
            "-i", input_path,
            "-ar", "44100",  # Taxa de amostragem de 44.1kHz
            "-ac", "1",      # Mono (1 canal)
            "-c:a", "pcm_s16le",  # Codec de áudio PCM 16-bit
            "-y",            # Sobrescrever arquivo de saída se existir
            output_path
        ]
        
        # Executar o comando
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Verificar se a conversão foi bem-sucedida
        if process.returncode != 0:
            error_message = process.stderr
            st.error(f"Erro na conversão com ffmpeg: {error_message}")
            
            # Tentar com pydub como alternativa
            try:
                audio = AudioSegment.from_file(input_path)
                audio = audio.set_channels(1)  # Converter para mono
                audio = audio.set_frame_rate(44100)  # Definir taxa de amostragem para 44.1kHz
                audio.export(output_path, format="wav")
            except Exception as pydub_err:
                st.error(f"Também falhou com pydub: {str(pydub_err)}")
                raise ValueError(f"Não foi possível converter o arquivo: {error_message}")
        
        return output_path
    
    except Exception as e:
        st.error(f"Erro ao tentar converter: {str(e)}")
        raise

def ensure_max_file_size(file_path, max_size_bytes=MAX_API_SIZE_BYTES):
    """Verifica se um arquivo está dentro do limite de tamanho e o reduz se necessário"""
    file_size = os.path.getsize(file_path)
    
    if file_size <= max_size_bytes:
        return file_path
    
    # Se estiver acima do limite, reduzir a qualidade
    try:
        audio = AudioSegment.from_wav(file_path)
        
        # Converter para mono se for estéreo
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # Reduzir a taxa de amostragem
        audio = audio.set_frame_rate(16000)  # Baixar para 16kHz
        
        # Exportar com compressão
        audio.export(file_path, format="wav", parameters=["-q:a", "3"])
        
        # Verificar novamente o tamanho
        new_size = os.path.getsize(file_path)
        if new_size <= max_size_bytes:
            return file_path
            
        # Se ainda estiver acima do limite, tentar compressão mais agressiva
        audio = audio.set_frame_rate(8000)  # Baixar para 8kHz
        audio.export(file_path, format="wav", parameters=["-q:a", "5"])
        
        return file_path
    except Exception as e:
        st.warning(f"Aviso: Não foi possível reduzir o tamanho do arquivo: {str(e)}")
        return file_path

def split_audio_file(file_path, segment_size_mb=MAX_SEGMENT_SIZE_MB, max_api_size=MAX_API_SIZE_BYTES):
    """
    Divide um arquivo de áudio em segmentos menores
    
    Args:
        file_path: Caminho para o arquivo de áudio
        segment_size_mb: Tamanho máximo de cada segmento em MB
        max_api_size: Tamanho máximo em bytes para a API
        
    Returns:
        Lista de caminhos para os arquivos de segmento
    """
    try:
        # Carregar o arquivo de áudio (agora sempre WAV)
        audio = AudioSegment.from_wav(file_path)
        
        # Calcular o número de segmentos necessários com base no tempo
        duration_ms = len(audio)
        
        # Calcular o tamanho esperado por minuto
        file_size = os.path.getsize(file_path)
        bytes_per_ms = file_size / duration_ms
        
        # Calcular o tamanho de segmento em ms, com margem de segurança de 20%
        segment_size_ms = int((max_api_size * 0.8) / bytes_per_ms)
        
        # Limitar o tamanho máximo do segmento (3-5 minutos é um bom tamanho)
        max_segment_ms = 3 * 60 * 1000  # 3 minutos
        segment_size_ms = min(segment_size_ms, max_segment_ms)
        
        # Garantir um tamanho mínimo razoável
        min_segment_ms = 30 * 1000  # 30 segundos
        segment_size_ms = max(segment_size_ms, min_segment_ms)
        
        # Calcular número de segmentos
        segments_count = math.ceil(duration_ms / segment_size_ms)
        
        # Criar lista para armazenar caminhos dos segmentos
        segment_paths = []
        
        # Dividir o áudio em segmentos
        for i in range(segments_count):
            start_ms = i * segment_size_ms
            end_ms = min((i + 1) * segment_size_ms, duration_ms)
            
            segment = audio[start_ms:end_ms]
            segment_path = f"{file_path}_segment_{i}.wav"
            
            # Exportar segmento com qualidade reduzida para garantir tamanho menor
            segment = segment.set_channels(1)  # Converter para mono
            segment = segment.set_frame_rate(16000)  # Reduzir para 16kHz
            segment.export(segment_path, format="wav")
            
            # Verificar tamanho do arquivo e reduzir ainda mais se necessário
            segment_path = ensure_max_file_size(segment_path)
            
            # Verificação final de tamanho
            if os.path.getsize(segment_path) > max_api_size:
                st.warning(f"Aviso: Segmento {i+1} ainda excede o limite de tamanho da API.")
                # Dividir mais uma vez se necessário
                mid_point = (end_ms - start_ms) // 2
                
                # Primeira metade
                segment1 = audio[start_ms:start_ms + mid_point]
                segment1_path = f"{file_path}_segment_{i}_part1.wav"
                segment1 = segment1.set_channels(1).set_frame_rate(16000)
                segment1.export(segment1_path, format="wav")
                segment1_path = ensure_max_file_size(segment1_path)
                segment_paths.append(segment1_path)
                
                # Segunda metade
                segment2 = audio[start_ms + mid_point:end_ms]
                segment2_path = f"{file_path}_segment_{i}_part2.wav"
                segment2 = segment2.set_channels(1).set_frame_rate(16000)
                segment2.export(segment2_path, format="wav")
                segment2_path = ensure_max_file_size(segment2_path)
                segment_paths.append(segment2_path)
                
                # Remover o segmento original grande
                try:
                    os.remove(segment_path)
                except:
                    pass
            else:
                segment_paths.append(segment_path)
        
        return segment_paths
    
    except Exception as e:
        st.error(f"Erro ao dividir arquivo: {str(e)}")
        
        # Método alternativo mais simples
        try:
            audio = AudioSegment.from_wav(file_path)
            duration_ms = len(audio)
            
            # Dividir em segmentos menores, 1 minuto
            segment_length_ms = 60 * 1000  # 1 minuto
            segments_count = math.ceil(duration_ms / segment_length_ms)
            
            segment_paths = []
            for i in range(segments_count):
                start_ms = i * segment_length_ms
                end_ms = min((i + 1) * segment_length_ms, duration_ms)
                
                segment = audio[start_ms:end_ms]
                segment_path = f"{file_path}_segment_{i}.wav"
                
                # Exportar com qualidade reduzida
                segment = segment.set_channels(1)
                segment = segment.set_frame_rate(16000)
                segment.export(segment_path, format="wav")
                
                # Verificação de tamanho
                segment_file_size = os.path.getsize(segment_path)
                if segment_file_size > max_api_size:
                    st.warning(f"Reduzindo qualidade do segmento {i+1} para cumprir limites da API")
                    segment = segment.set_frame_rate(8000)
                    segment.export(segment_path, format="wav")
                
                segment_paths.append(segment_path)
            
            return segment_paths
            
        except Exception as alt_err:
            st.error(f"Método alternativo de segmentação também falhou: {str(alt_err)}")
            raise

def transcribe_segment(segment_path, client, language):
    """
    Transcreve um segmento de áudio usando a API OpenAI
    
    Args:
        segment_path: Caminho para o arquivo de segmento
        client: Cliente OpenAI inicializado
        language: Código do idioma
        
    Returns:
        Texto transcrito
    """
    # Verificar tamanho do arquivo
    file_size = os.path.getsize(segment_path)
    
    if file_size > MAX_API_SIZE_BYTES:
        st.warning(f"Aviso: Segmento {segment_path} tem {file_size/BYTES_PER_MB:.2f}MB (acima do limite). Tentando reduzir.")
        segment_path = ensure_max_file_size(segment_path)
    
    with open(segment_path, "rb") as audio_file:
        try:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=language
            )
            return transcript.text
        except Exception as e:
            error_msg = str(e)
            if "413" in error_msg:
                st.error(f"Erro: O segmento excede o limite máximo de tamanho da API (25MB). Tente com um arquivo menor.")
                st.error(f"Tamanho do segmento: {file_size/BYTES_PER_MB:.2f}MB")
                return f"[Erro de tamanho no segmento - transcription falhou] "
            else:
                st.error(f"Erro ao transcrever segmento: {error_msg}")
                return f"[Erro na transcrição deste segmento] "

# Verificar FFMPEG
if not check_ffmpeg_installed():
    st.warning("⚠️ FFMPEG não encontrado ou não está funcionando corretamente. A conversão de formatos pode falhar.")

# Quando o usuário clicar no botão de transcrição
if st.button("Transcrever") and uploaded_file is not None:
    # Verificar o tamanho do arquivo
    file_size_mb = uploaded_file.size / BYTES_PER_MB
    
    if file_size_mb > MAX_UPLOAD_SIZE_MB:
        st.error(f"O arquivo é muito grande! O tamanho máximo permitido é {MAX_UPLOAD_SIZE_MB}MB.")
        st.stop()
    
    # Criar uma barra de progresso
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Configurar cliente OpenAI
        client = OpenAI(api_key=api_key)
        
        # Criar diretório temporário para os arquivos
        with tempfile.TemporaryDirectory() as temp_dir:
            # Salvar o arquivo temporariamente
            original_file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(original_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Obter formato do arquivo
            file_format = os.path.splitext(original_file_path)[1].lower()[1:]
            
            status_text.text("Preparando arquivo de áudio...")
            progress_bar.progress(0.1)
            
            # Converter para WAV se não for WAV
            if file_format.lower() != "wav":
                status_text.text(f"Convertendo arquivo {file_format} para WAV...")
                try:
                    wav_file_path = convert_audio_to_wav(original_file_path, temp_dir)
                    status_text.text("Conversão para WAV concluída!")
                except Exception as conv_err:
                    st.error(f"Falha ao converter o arquivo: {str(conv_err)}")
                    st.stop()
            else:
                wav_file_path = original_file_path
            
            progress_bar.progress(0.2)
            
            # Se o arquivo for menor que o limite da API, processar diretamente
            wav_file_size = os.path.getsize(wav_file_path) / BYTES_PER_MB
            if wav_file_size <= MAX_SEGMENT_SIZE_MB:
                status_text.text("Transcrevendo arquivo (único segmento)...")
                progress_bar.progress(0.3)
                
                # Verificar e garantir que o arquivo está dentro do limite da API
                wav_file_path = ensure_max_file_size(wav_file_path)
                
                with open(wav_file_path, "rb") as audio_file:
                    file_size = os.path.getsize(wav_file_path)
                    if file_size > MAX_API_SIZE_BYTES:
                        st.error(f"Arquivo ainda está muito grande: {file_size/BYTES_PER_MB:.2f}MB")
                        st.stop()
                        
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language=idioma
                    )
                
                full_transcript = transcript.text
                progress_bar.progress(0.9)
            else:
                # Dividir o arquivo em segmentos
                status_text.text("Dividindo arquivo em segmentos...")
                progress_bar.progress(0.3)
                
                try:
                    segment_paths = split_audio_file(wav_file_path)
                    total_segments = len(segment_paths)
                    
                    # Verificar tamanho de cada segmento
                    for i, segment_path in enumerate(segment_paths):
                        segment_size = os.path.getsize(segment_path) / BYTES_PER_MB
                        if segment_size > 24:  # 24MB é um limite seguro
                            st.warning(f"Segmento {i+1} tem {segment_size:.2f}MB (próximo ao limite).")
                    
                    status_text.text(f"Arquivo dividido em {total_segments} segmentos. Iniciando transcrição...")
                    
                    # Transcrever cada segmento
                    full_transcript = ""
                    for i, segment_path in enumerate(segment_paths):
                        progress_percent = 0.3 + (i / total_segments) * 0.6
                        progress_bar.progress(progress_percent)
                        status_text.text(f"Transcrevendo segmento {i+1} de {total_segments}...")
                        
                        segment_transcript = transcribe_segment(segment_path, client, idioma)
                        full_transcript += segment_transcript + " "
                        
                        # Remover arquivo do segmento
                        try:
                            os.remove(segment_path)
                        except:
                            pass  # Ignorar erros na remoção
                except Exception as e:
                    st.error(f"Erro ao processar os segmentos: {str(e)}")
                    st.stop()
            
            # Limpar e formatar a transcrição final
            full_transcript = full_transcript.strip()
            
            # Atualizar progresso
            progress_bar.progress(1.0)
            status_text.text("Transcrição concluída!")
            
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
            
    except Exception as e:
        st.error(f"Ocorreu um erro durante a transcrição: {str(e)}")
        st.info("""
        Se você encontrou um erro:
        1. Tente fazer upload de um arquivo menor ou de menor qualidade
        2. Verifique se o arquivo não está corrompido
        3. Tente converter o arquivo para MP3 de baixa qualidade antes de fazer upload
        """)

# Adicionar instruções e informações adicionais
with st.expander("Como funciona o processamento de arquivos?"):
    st.markdown("""
    ### Processo de conversão e transcrição:

    1. **Upload**: Você faz upload de qualquer arquivo de áudio suportado
    2. **Conversão**: O sistema converte automaticamente para WAV, se necessário
    3. **Divisão**: Para arquivos maiores que 20MB, o sistema divide em segmentos menores
    4. **Otimização**: Cada segmento é otimizado para ficar abaixo do limite de 25MB da API
    5. **Processamento**: Cada segmento é enviado para a API do Whisper
    6. **Combinação**: As transcrições de todos os segmentos são combinadas
    7. **Resultado**: Você recebe a transcrição completa para download

    ### Limites e otimizações:
    
    - A API Whisper da OpenAI tem um limite de 25MB por arquivo
    - O sistema reduz automaticamente a qualidade do áudio se necessário
    - Arquivos muito grandes são divididos em segmentos menores
    - Cada segmento é otimizado para garantir compatibilidade com a API
    """)

# Instruções e informações adicionais
st.markdown("---")
st.markdown("""
### Como usar:
1. Faça upload de um arquivo de áudio (até 200MB)
2. Selecione o idioma do áudio
3. Clique em "Transcrever"
4. Acompanhe o progresso da conversão e transcrição
5. Baixe o resultado como arquivo TXT

### Dicas para melhor desempenho:
- Arquivos menores são processados mais rapidamente
- Áudio com boa qualidade de gravação resulta em melhor transcrição
- Para arquivos muito grandes, considere dividi-los antes do upload
""")

# Rodapé
st.markdown("---")
st.markdown("Desenvolvido com Streamlit e OpenAI Whisper API | Suporta múltiplos formatos de áudio")
