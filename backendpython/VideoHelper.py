import os
import tempfile
import requests
import openai
import subprocess
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
import tiktoken
import re

openai.api_key = os.getenv('OPENAI_API_KEY')

def get_captions(url):
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, '%(id)s.%(ext)s')
    cmd = [
        'yt-dlp',
        '--skip-download',
        '--write-auto-sub',
        '--sub-lang', 'en',
        '--output', output_path,
        url
    ]
    subprocess.run(cmd, check=False)
    for file in os.listdir(temp_dir):
        if file.endswith('.vtt') or file.endswith('.srt'):
            caption_file = os.path.join(temp_dir, file)
            with open(caption_file, 'r', encoding='utf-8') as f:
                content = f.read()
            return vtt_srt_to_text(content)
    return None

def vtt_srt_to_text(content):
    lines = content.splitlines()
    text_lines = []
    for line in lines:
        if line.strip() == '' or '-->' in line or line.strip().isdigit():
            continue
        text_lines.append(line)
    return ' '.join(text_lines)

def download_video(url):
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, 'video.mp4')
    cmd = [
        'yt-dlp',
        '-f', 'bestvideo+bestaudio/best',
        '-o', video_path,
        url
    ]
    subprocess.run(cmd, check=True)
    return video_path

def transcribe_with_whisper(video_path):
    import whisper
    model = whisper.load_model('base')
    result = model.transcribe(video_path)
    return result['text']

def chunk_text(text, max_tokens=3000):
    # Use tiktoken to count tokens if available, else fallback to word count
    try:
        enc = tiktoken.encoding_for_model("gpt-4")
        tokens = enc.encode(text)
        if len(tokens) <= max_tokens:
            return [text]
        # Split by sentences for better chunking
        sentences = re.split(r'(?<=[.!?]) +', text)
        chunks = []
        current = ""
        for sentence in sentences:
            if len(enc.encode(current + sentence)) > max_tokens:
                if current:
                    chunks.append(current.strip())
                current = sentence + " "
            else:
                current += sentence + " "
        if current:
            chunks.append(current.strip())
        return chunks
    except Exception:
        # Fallback: split by words
        words = text.split()
        chunks = [" ".join(words[i:i+max_tokens]) for i in range(0, len(words), max_tokens)]
        return chunks


def recursive_summarize(text, llm, prompt, max_tokens=3000):
    chunks = chunk_text(text, max_tokens=max_tokens)
    if len(chunks) == 1:
        return LLMChain(llm=llm, prompt=prompt).invoke({"transcript": chunks[0]})["text"]
    chunk_summaries = []
    for chunk in chunks:
        summary = LLMChain(llm=llm, prompt=prompt).invoke({"transcript": chunk})["text"]
        chunk_summaries.append(summary)
    combined = " ".join(chunk_summaries)
    # If combined is still too long, recurse
    enc = None
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4")
        if len(enc.encode(combined)) > max_tokens:
            return recursive_summarize(combined, llm, prompt, max_tokens)
    except Exception:
        if len(combined.split()) > max_tokens:
            return recursive_summarize(combined, llm, prompt, max_tokens)
    return LLMChain(llm=llm, prompt=prompt).invoke({"transcript": combined})["text"]


def summarize_and_tips(transcript, title="", description=""):
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    # Truncate transcript if too long (approx 6000 tokens for safety)
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4")
        tokens = enc.encode(str(transcript))
        if len(tokens) > 6000:
            transcript = enc.decode(tokens[:6000])
    except Exception:
        words = str(transcript).split()
        if len(words) > 4000:
            transcript = " ".join(words[:4000])
    prompt = f"""Please provide a comprehensive analysis of the following video content:\n\nVideo Title: {title}\nVideo Description: {description}\nTranscript: {transcript}\n\nPlease provide:\n1. A 5-sentence summary of the main content\n2. A list of 5-7 practical tips or key insights\n3. The main category and subcategory this content belongs to\n\nFormat your response as JSON:\n{{\n  \"summary\": \"5-sentence summary here\",\n  \"tips\": [\"tip1\", \"tip2\", \"tip3\", \"tip4\", \"tip5\"],\n  \"category\": \"main category\",\n  \"subcategory\": \"subcategory\",\n  \"keyPoints\": [\"key point 1\", \"key point 2\", \"key point 3\"]\n}}"""
    response = llm.invoke(prompt)
    response_text = response.content if hasattr(response, 'content') else str(response)
    import re, json
    try:
        if not isinstance(response_text, str):
            response_text = str(response_text)
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            summary_data = json.loads(json_match.group(0))
        else:
            raise ValueError('No JSON object found in GPT response')
    except Exception:
        summary_data = {
            "summary": response_text,
            "tips": ["Key insight from the video"],
            "category": "General",
            "subcategory": "Education",
            "keyPoints": ["Main point from the video"]
        }
    return summary_data["summary"], summary_data["tips"]


def generate_embeddings(text):
    embeddings = OpenAIEmbeddings()
    return embeddings.embed_documents([text])[0]

def process_video(url):
    transcript = get_captions(url)
    title = ""
    description = ""
    if not transcript:
        video_path = download_video(url)
        transcript = transcribe_with_whisper(video_path)
    # Optionally, fetch title/description using yt-dlp for better parity
    try:
        import yt_dlp
        with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
            info = ydl.extract_info(url, download=False)
            if info is not None:
                title = info.get("title", "")
                description = info.get("description", "")
    except Exception:
        pass
    summary, tips = summarize_and_tips(transcript, title, description)
    embedding = generate_embeddings(transcript)
    return {
        'transcript': transcript,
        'summary': summary,
        'tips': tips,
        'embedding': embedding
    } 