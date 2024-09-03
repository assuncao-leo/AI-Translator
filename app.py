import streamlit as st
import openai
import pandas as pd
from openai import OpenAI
from streamlit import image
import os
import tempfile

if __name__ == '__main__':

# App title
    st.set_page_config(
        page_title="AI-SRT Translator",
        page_icon=":clipboard:",
        layout="wide",
        initial_sidebar_state="auto",
    )

st.title('Welcome to the AI-SRT Translator App!⚡')
st.markdown(
        "Welcome to the AI-SRT Translator App! Drop your SRT or TXT file below, select the parameters desired on the left menu, and let the AI translate the subtitles for you!"
    )

with st.sidebar:
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
    prompt=f"Translate the following {source_language} subtitle text to {target_language}, keeping the same language sentiment and specific currencies and personal names in the source language (i.e. money currencies, city names, etc), and converting English expressions to the equivalent of the output language (when applicable).The text is:\n\n{text}"
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
            # Skip sections that don't have enough lines
            updated_sections.append(section)
            continue

        section_number = lines[0].strip()
        time_info = lines[1].strip()
        text = ' '.join(lines[2:]).strip()  # Join all text lines in the section

        split_lines = split_text_to_lines(text, max_chars_per_line)

        # Ensure the number of lines does not exceed max_lines_per_section
        if len(split_lines) > max_lines_per_section:
            split_lines = split_lines[:max_lines_per_section]

        updated_section = f"{section_number}\n{time_info}\n" + '\n'.join(split_lines)
        updated_sections.append(updated_section)

    return '\n\n'.join(updated_sections)

def translate_srt_file(temp_file_path, output_file_path, batch_size=30, max_chars_per_line=char_per_line, max_lines_per_section=lines_per_section):
    srt_content = read_srt_file(temp_file_path)
    subtitle_sections = split_into_sections(srt_content)

    # Ensure the output file is empty before starting
    write_srt_file(output_file_path, "", mode='w')

    for i in range(0, len(subtitle_sections), batch_size):
        batch = subtitle_sections[i:i + batch_size]

        translated_batch = []
        for section in batch:
            try:
                # Split section into header and body
                header, body = section.split('\n', 2)[0:2], section.split('\n', 2)[2]
            except ValueError:
                translated_batch.append(section)
                continue

            translated_body = translate_text(body)
            translated_section = f"{header[0]}\n{header[1]}\n{translated_body}"
            translated_batch.append(translated_section)

        # Apply SRT rules before writing to the output file
        updated_content = enforce_srt_rules("\n\n".join(translated_batch), max_chars_per_line, max_lines_per_section)

        # Write the translated and updated batch to the output file
        write_srt_file(output_file_path, updated_content + "\n\n", mode='a')

uploaded_file = st.file_uploader("Choose a SRT file", type=["srt", "txt"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".srt") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_file_path = tmp_file.name
    st.success('SRT file loaded successfully!', icon='✅')
    generate_button = st.button("Translate Subtitles")

    if generate_button:
        try:
            if not openai_key:
              st.error("API key is missing. Please insert your API key before proceeding.")
            else:
                with st.spinner("Translating the file. It could take a few minutes."):
                    output_file_path = 'translated_file.srt'
                    translate_srt_file(temp_file_path, output_file_path)
                    st.success('Translation is completed!', icon='✅')
                    with open(output_file_path, 'rb') as file:
                        st.download_button(
                            label="Download Translated File",
                            data=file,
                            file_name="translated_file.srt",
                            mime="text/plain"
                        )

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
            st.error("An error occurred while translating the file. Please try again.")
            st.write(e)
else:
    st.info("Please upload a SRT or TXT file to get started.")
