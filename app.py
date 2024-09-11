import streamlit as st
import openai
import pandas as pd
from openai import OpenAI
from streamlit import image
import os
import tempfile
from pydub import AudioSegment
import re

if __name__ == '__main__':

# app title
    st.set_page_config(
        page_title="AI-SRT Translator",
        page_icon=":clipboard:",
        layout="wide",
        initial_sidebar_state="auto",
    )

st.title('Welcome to the AI-SRT Translator App!:clipboard:')
st.subheader("Created by: Leonardo Assunção")
st.markdown(
        "Simply drop your SRT or TXT file below, select the parameters desired on the left menu, and let the AI translate the subtitles for you!"
    )

with st.sidebar:
    image("translator_icon.png", width=300)
    openai_key = st.text_input(label="Enter your OpenAI API key: [(click here to obtain a new key if you don't have one)](https://platform.openai.com/account/api-keys)",
                type="password", help="Your API key is not stored anywhere")
    if openai_key:
        st.success('API key loaded successfully!', icon='✅')
        open_ai_key = openai_key
        client = OpenAI(api_key=open_ai_key)
    llm_model = st.selectbox(label="Choose a model", options=["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4", "gpt-4o"])
    source_language = st.selectbox(label="Select the current language of the file", options = [
    "English",
    "Mandarin Chinese",
    "Spanish",
    "Arabic",
    "Hindi",
    "Portuguese",
    "Brazilian Portuguese",
    "Japanese",
    "French",
    "German",
    "Russian",
    "Korean",
    "Romanian",
    "Indonesian",
    "Italian",
    "Turkish",
    "Thai"
], index=0,)
    target_language = st.selectbox(label="Select the output language you want to translate the file", options = [
    "English",
    "Mandarin Chinese",
    "Spanish",
    "Arabic",
    "Hindi",
    "Portuguese",
    "Brazilian Portuguese",
    "Japanese",
    "French",
    "German",
    "Russian",
    "Korean",
    "Romanian",
    "Indonesian",
    "Italian",
    "Turkish",
    "Thai"
])
    char_per_line = st.selectbox(label="Select the max number of characters per line", options = list(range(15, 61)), index=21,)
    lines_per_section = st.selectbox(label="Select the max number of lines", options = list(range(1, 5)), index=1,)

def translate_text(text):

    model_name = llm_model
    prompt=f"Your task is to translate the indicated {source_language} subtitle text to {target_language}, keeping the same language sentiment and specific currencies and personal names in the source language (i.e. money currencies, city names, etc), and converting English expressions to the equivalent of the output language (when applicable). Your output answer should contain ONLY the translated text, nothing else. The text is:\n\n{text}"
    response = client.chat.completions.create(
        model=model_name,
        max_tokens=None,
        temperature=0.2,
        stop=None,
        messages=[
            {"role": "system", "content": prompt}
        ]
    )

    translated_text = response.choices[0].message.content.strip()
    return translated_text

def read_srt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

def write_srt_file(file_path, content, mode='w'):
    with open(file_path, mode, encoding='utf-8') as file:
        file.write(content)

def split_into_sections(srt_content):
    sections = srt_content.strip().split('\n\n')
    return sections

def split_text_to_lines(text, max_chars_per_line):
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        if sum(len(w) for w in current_line) + len(current_line) + len(word) <= max_chars_per_line:
            current_line.append(word)
        else:
            lines.append(' '.join(current_line))
            current_line = [word]

    if current_line:
        lines.append(' '.join(current_line))

    return lines

def enforce_srt_rules(content, max_chars_per_line, max_lines_per_section):
    sections = content.strip().split('\n\n')
    updated_sections = []

    for section in sections:
        lines = section.split('\n')

        if len(lines) < 3:
            # to skip sections without have enough lines
            updated_sections.append(section)
            continue

        section_number = lines[0].strip()
        time_info = lines[1].strip()
        text = ' '.join(lines[2:]).strip()  # Join all text lines in the section

        split_lines = split_text_to_lines(text, max_chars_per_line)

        if len(split_lines) > max_lines_per_section:
            split_lines = split_lines[:max_lines_per_section]

        updated_section = f"{section_number}\n{time_info}\n" + '\n'.join(split_lines)
        updated_sections.append(updated_section)

    return '\n\n'.join(updated_sections)

def translate_srt_file(temp_file_path, output_file_path, batch_size=30, max_chars_per_line=char_per_line, max_lines_per_section=lines_per_section):
    srt_content = read_srt_file(temp_file_path)
    subtitle_sections = split_into_sections(srt_content)

    write_srt_file(output_file_path, "", mode='w')

    for i in range(0, len(subtitle_sections), batch_size):
        batch = subtitle_sections[i:i + batch_size]

        translated_batch = []
        for section in batch:
            try:
                # split the sections in header/body
                header, body = section.split('\n', 2)[0:2], section.split('\n', 2)[2]
            except ValueError:
                translated_batch.append(section)
                continue

            translated_body = translate_text(body)
            translated_section = f"{header[0]}\n{header[1]}\n{translated_body}"
            translated_batch.append(translated_section)

        # apply the SRT rules before writing to the output file
        updated_content = enforce_srt_rules("\n\n".join(translated_batch), max_chars_per_line, max_lines_per_section)

        # write the translated and updated batch to the output file
        write_srt_file(output_file_path, updated_content + "\n\n", mode='a')
################################################################################################  Audio functions ###################################################
def format_time_srt(seconds):
    """Convert seconds to SRT time format (hh:mm:ss,ms)."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{milliseconds:03}"

def time_to_seconds(srt_time):
    """Convert SRT time format (hh:mm:ss,ms) to seconds."""
    hours, minutes, seconds, milliseconds = map(int, re.split('[:,]', srt_time))
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000

def transcribe_audio_with_timestamp_continuity(audio_file_path, srt_file_path, segment_duration_sec=600):
    audio = AudioSegment.from_file(audio_file_path)
    segment_duration_ms = segment_duration_sec * 1000
    with open(srt_file_path, 'w', encoding='utf-8') as srt_file:
        idx = 1  # Global section number
        start_time_ms = 0  # Start from the beginning of the audio
        while start_time_ms < len(audio):
            end_time_ms = min(start_time_ms + segment_duration_ms, len(audio))
            segment = audio[start_time_ms:end_time_ms]
            segment_file_path = "temp_segment.mp3"
            segment.export(segment_file_path, format="mp3")

            with open(segment_file_path, "rb") as audio_file:
                try:
                    transcription_response = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="srt"
                    )
                except Exception as e:
                    raise ValueError(f"Failed to transcribe audio: {e}")

            for line in transcription_response.splitlines():
                if "-->" in line:  
                    start_srt_time, end_srt_time = line.split(" --> ")
                    start_seconds = time_to_seconds(start_srt_time)
                    end_seconds = time_to_seconds(end_srt_time)

                    adjusted_start_time = start_time_ms / 1000 + start_seconds
                    adjusted_end_time = start_time_ms / 1000 + end_seconds

                    srt_file.write(f"{idx}\n")  
                    srt_file.write(f"{format_time_srt(adjusted_start_time)} --> {format_time_srt(adjusted_end_time)}\n")
                    idx += 1
                elif not line.isdigit():  
                    srt_file.write(line + "\n")

            start_time_ms = end_time_ms  
#####################################################################################

st.markdown("<h2 style='text-align: left; color: #333;'>Translate your SRT file</h2>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose a SRT file", type=["srt", "txt"])

st.markdown("<h2 style='text-align: left; color: #333;'>Generate subtitles for your audio file</h2>", unsafe_allow_html=True)
uploaded_file_2 = st.file_uploader("Choose an audio file", type=["mp3", "wav", "mp4", "m4a", "acc", "webm", "mpeg", "flac", "ogg"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".srt") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_file_path = tmp_file.name
    st.success('SRT file loaded successfully!', icon='✅')
    generate_button = st.button("Translate Subtitles")

    if generate_button:
            if not openai_key:
              st.error("API key is missing. Please insert your API key before proceeding.")
            else:
                with st.spinner("Translating the file. It could take a few minutes."):
                    output_file_path = 'translated_file.srt'
                    try:
                        translate = translate_srt_file(temp_file_path, output_file_path)
                        with open(output_file_path, 'rb') as file:
                            st.session_state['file_data'] = file.read()
                        st.success('Translation is completed!', icon='✅')
                    except openai.RateLimitError as e:
                        st.markdown(
                            "It looks like you do not have OpenAI API credits left. Check [OpenAI's usage webpage for more information](https://platform.openai.com/account/usage)"
                        )
                        st.write(e)
                    except openai.NotFoundError as e:
                        st.warning(
                            "It looks like you do not have entered you Credit Card information on OpenAI's site. Buy pre-paid credits to use the API and try again.",
                            icon="💳"
                        )
                        st.write(e)
                    except Exception as e:
                        st.write(f"An error occurred while translating the file: {e}")

if 'file_data' in st.session_state:
    st.download_button(
        label="Download Translated File",
        data=st.session_state['file_data'],
        file_name="translated_file.srt",
        mime="text/plain"
    )

### audio button
if uploaded_file_2 is not None:
    audio_file_path = uploaded_file_2
    st.success('audio file loaded successfully!', icon='✅')
    generate_button = st.button("Generate SRT file with Subtitles")

    if generate_button:
            if not openai_key:
              st.error("API key is missing. Please insert your API key before proceeding.")
            else:
                with st.spinner("Generating the SRT file. It could take a few minutes."):
                    srt_file_path = 'output_audio_srt.srt'
                    try:
                        transcribe_audio_with_timestamp_continuity(audio_file_path, srt_file_path, segment_duration_sec=600)
                        output_file_path = 'translated_file.srt'
              
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".srt") as tmp_file:
                            with open(srt_file_path, 'r', encoding='utf-8') as srt_file:
                                srt_content = srt_file.read()
                            tmp_file.write(srt_content.encode('utf-8'))
                            temp_file_path = tmp_file.name
                        translate = translate_srt_file(temp_file_path, output_file_path)

                        with open(output_file_path, 'rb') as file:
                           st.session_state['file_data_2'] = file.read()
                        st.success('Your SRT file has been generated successfully!', icon='✅')
                    except openai.RateLimitError as e:
                        st.markdown(
                            "It looks like you do not have OpenAI API credits left. Check [OpenAI's usage webpage for more information](https://platform.openai.com/account/usage)"
                        )
                        st.write(e)
                    except openai.NotFoundError as e:
                        st.warning(
                            "It looks like you do not have entered you Credit Card information on OpenAI's site. Buy pre-paid credits to use the API and try again.",
                            icon="💳"
                        )
                        st.write(e)
                    except Exception as e:
                        st.write(f"An error occurred while translating the file: {e}")
if 'file_data_2' in st.session_state:
    st.download_button(
        label="Download SRT File",
        data=st.session_state['file_data_2'],
        file_name="subtitles_file.srt",
        mime="text/plain"
      )
