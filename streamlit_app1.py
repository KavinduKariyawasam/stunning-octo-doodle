import streamlit as st
import shutil
import torch
import replicate
import base64
import streamlit as st
import os, sys
import io 
import requests
from time import strftime
from pydub import AudioSegment
from io import BytesIO
from audio_recorder_streamlit import audio_recorder
from streamlit_mic_recorder import mic_recorder,speech_to_text
from huggingface_hub import snapshot_download
import time

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path
from generate_videos import generate_videos
from voices import voice_dict
import google.generativeai as genai
#from transformers import AutoTokenizer, AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers import AutoTokenizer, AutoModelForSpeechSeq2Seq, AutoProcessor
import transformers


from st_clickable_images import clickable_images
from PIL import Image


#-------------------------------------Whisper model (new-api)------------------------------------------------------------#
def whisper_model(audio):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(                        ## There is an error in importing pipeline from transformers
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    result = pipe(audio)
    return (result["text"])


#--------------------------------------- Download the pre-trained model (Not using right now)--------------------------------#
def download_model():
    REPO_ID = 'vinthony/SadTalker-V002rc'
    snapshot_download(repo_id=REPO_ID, local_dir='./checkpoints', local_dir_use_symlinks=True)


#------------------------------------- Function to play the video and record audio -----------------------------------------#
def play_video(uploaded_audio_files):
    ''' Input : list containing the path for input audio files
        Output: '''

    uploaded_audio_files = uploaded_audio_files
    result_dir = "/content/drive/MyDrive/Streamlit/results"
    st.markdown(
        f'<style>video {{ width: 100%; max-width: 325px; height: 325px; }}</style>',
        unsafe_allow_html=True
    )

    if st.sidebar.button("Play Previous Video") and st.session_state.video_index > 0:
        st.session_state.video_index -= 1

    if st.sidebar.button("Play Next Video") and st.session_state.video_index < len(uploaded_audio_files) - 1:
        st.session_state.video_index += 1
    elif st.session_state.video_index > len(uploaded_audio_files):
        st.balloons()
        st.write(st.session_state.video_index)

    if st.session_state.video_index < len(uploaded_audio_files):
        st.markdown(f"Question {st.session_state.video_index + 1}")
        video_path = open(os.path.join(result_dir, f"output_{st.session_state.video_index}.mp4"), 'rb')

        video = st.video(video_path, format='video/mp4')
        
        audio_or_text = st.selectbox("Select the output format:", ["Audio file", "Text"])

        if audio_or_text == "Audio file":
            c1,c2, c3=st.columns(3)
            with c1:
              st.write(f"Click to record your voice for Question {st.session_state.video_index + 1}:")

            # Check if audio has been recorded for the current video
            audio_key = f'audio_{st.session_state.video_index}'
            if audio_key not in st.session_state:
                with c2:
                    audio = mic_recorder(start_prompt="Start recording answer", stop_prompt="Stop", key='recorder')


                if st.button("Save answer", type="primary"):
                    audio_filename = f"answer_{st.session_state.video_index}.mp3"
                    audio_path = os.path.join(result_dir, audio_filename)
                    with open(audio_path, 'wb') as audio_file:
                        audio_file.write(audio['bytes'])
                    st.success("Saving process successfully completed..")

                if st.button("View recorded answer"):
                    audio_filename = f"answer_{st.session_state.video_index}.mp3"
                    audio_path = os.path.join(result_dir, audio_filename)
                    st.info("Text format:")
                    st.write(whisper_model(audio_path))
                    st.info("Audio format:")
                    st.audio(audio_path)
                
                    
            else:
                st.write(f"Audio already recorded for Question {st.session_state.video_index + 1}.")
        
        
        if audio_or_text == "Text":
            
            state=st.session_state

            if 'text_received' not in state:
                state.text_received=[]

            c1,c2=st.columns(2)
            with c1:
                st.write("Convert speech to text:")
            with c2:
                text=speech_to_text(language='en',use_container_width=True,just_once=True,key='STT')
              
            if text and (len(state.text_received) >= st.session_state.video_index+1):  
                state.text_received[st.session_state.video_index] = text
            elif text is not None:
                state.text_received.append(text)

            #for text in state.text_received:
            #    st.text(text)
            if len(state.text_received) >= st.session_state.video_index + 1:
                st.info("Recorded text")
                st.text(state.text_received[st.session_state.video_index])
                st.write(state.text_received)

            if st.sidebar.button("Reset"):
                state.text_received=[]
                # Reset the video index and remove all recorded audio files
                #st.session_state.video_index = 0
                #st.session_state.uploaded_audio_files = []
                #for i in range(len(uploaded_audio_files)):
                #    audio_key = f'audio_{i}'
                #   if audio_key in st.session_state:
                  #      del st.session_state[audio_key]


            

    elif len(uploaded_audio_files) == 0:
        st.warning("Please generate the videos first!")
    else:
        st.balloons()
    
    

#----------------------------------------- Text to voice with elevenlabs api---------------------------------------------------#
def text_to_speech(text, voice_id):
    ''' Input : Text that should convert to the voice
                Voice id (This can be selected using a select box)

        Output:  Audio file that was generated using the text
    '''

    CHUNK_SIZE = 1024
    url = "https://api.elevenlabs.io/v1/text-to-speech/" + voice_id

    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": "39d2bd5344d7cc5b29a1b5cfd515a745"   # Elevenlabs api
    }

    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }

    response = requests.post(url, json=data, headers=headers)
    return response.content

#------------------------------------Speech to text with whisper-large-v3 by openai (Not using)----------------------------------------------#
def speech2text(filename):
    ''' Input:  Audio file path
        Output: json containing the text output
    '''
    #st.audio(filename)
    API_TOKEN = "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"
    API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    #with open(filename, "rb") as f:
    #    data = f.read()
    audio_content = filename.read()
    response = requests.post(API_URL, headers=headers, data=audio_content)
    return response.json()


#----------------------------- LLM models to test -----------------------------------------------#

#LLM llama v2 model for generating answers
def llama(prompt):
    ''' Input : Prompt 
        Output: Generated answer from the llama2 LLM

        Note : This is an api from replicate
    '''

    os.environ["REPLICATE_API_TOKEN"] = "r8_CD5rbfDS9Tql0Nh9aK94n8KCoIpWMaP3WHrCJ"   # Replicate api token
    api = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"])
    output = api.run(
        "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
            input={"prompt": prompt + ". Don't use more than 40 words to explain."}
        )

    # An empty string to hold the generated answer
    ans_from_llama = ''

    for item in output:
        ns_from_llama += item

    return ans_from_llama

# google gemini pro model
def gemini_pro(prompt):
    ''' Input : Prompt 
        Output: Generated answer from the gemini-pro model
    '''
    genai.configure(api_key="")   # google api
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt + ". Use only 30 words to explain.")
    return response.text


#---------------------------------------Function to autoplay audio files (Not using right now) --------------------------------------------
def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )

#-----------------------------------------------------------------------------------------------------#
def ask_the_agent(uploaded_image):
    result_dir = "/content/drive/MyDrive/Streamlit/results"
    user_question = mic_recorder(start_prompt="Start talking", stop_prompt="Stop", key='recorder')

    if user_question:
        #st.audio(user_question['bytes'], format="audio/wav")
        audio_file = io.BytesIO(user_question['bytes'])
        question = whisper_model(user_question['bytes'])
        st.info(f"Question: {question}")
        #st.write(question)

        #st.audio(audio, format="audio/mp3")

        if st.button("Ask the bot", type='primary'): 
            llm_output = gemini_pro(question)
            st.info(f"LLM Output: {llm_output}")
            audio = text_to_speech(llm_output, "29vD33N1CtxCmqQRPOHJ")
            mp3_file_path = os.path.join(result_dir, f"answer.mp3")
            if len(st.session_state.voice_answer) >= 1:
                st.session_state.voice_answer = []
            st.session_state.voice_answer.append(mp3_file_path)
            audio_segment = AudioSegment.from_file(BytesIO(audio), format="mp3")
            audio_segment.export(mp3_file_path, format="mp3")       
            generate_videos('extcrop', True, 0, uploaded_image, st.session_state.voice_answer)
            st.markdown(
                f'<style>video {{ width: 100%; max-width: 300px; height: 300px; }}</style>',
                unsafe_allow_html=True
            )
            st.video("/content/drive/MyDrive/Streamlit/results/output_0.mp4")


#--------------------------------------------- Main function ---------------------------------------------------#
# Main function for the Streamlit app
def main():
    st.set_page_config(
        page_title='Avatar Creation DEMO',
        page_icon='😎',
        layout='wide',
        initial_sidebar_state='expanded'
    )

    st.title(":blue[Avatar Creation DEMO]")

    #download_model()

    # Create or get SessionState
    if 'uploaded_audio_files' not in st.session_state:
        st.session_state.uploaded_audio_files = []
        st.session_state.video_index = 0
    
    if 'voice_answer' not in st.session_state:
        st.session_state.voice_answer = []
    
    # Page navigation 
    rad = st.sidebar.radio("Page Navigation", ["Generate Videos", "Play Videos","Any questions!"])
    
    # Generate videos page
    if rad == "Generate Videos":
        st.header("Generate Videos")

        file_ = open("/content/drive/MyDrive/Streamlit/ai-avatar-free-box.gif", "rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()

        st.markdown(
            f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
            unsafe_allow_html=True,
        )
        ###############Clickable default images###############
        # Convert the single image to base64
        #with open("/content/man_new.jpg", "rb") as image:
        #    encoded = base64.b64encode(image.read()).decode()
        #    image_path = f"data:image/jpeg;base64,{encoded}"

        # Show only one image
        #clicked = clickable_images(
        #    [image_path],
        #    titles=["Man Image"],
        #    div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
        #    img_style={"margin": "5px", "height": "200px"},
        #)
        
        #if clicked == 0:
        #    image_path = "/content/man_new.jpg"
        #    with open(image_path, "rb") as image_file:
        #        uploaded_image = Image.open(image_path)
        #else:
        #    uploaded_image = st.file_uploader("Upload Image HERE", type=["jpg", "jpeg", "png"])
            ##############################



        

        st.sidebar.header("Audio settings")
        voice_type = st.sidebar.selectbox('Select the gender:', ['Male', 'Female'])

        if voice_type == 'Male':
            voice_selected = st.sidebar.selectbox("Select the voice type:", [
                'Drew', 'Clyde', 'Paul', 'Dave', 'Fin', 'Antoni', 'Thomas', 'Charlie', 'George', 'Callum', 
                'Patrick', 'Harry', 'Liam', 'Josh', 'Arnold', 'Matthew', 'James', 'Joseph', 'Jeremy', 'Michael', 
                'Ethan', 'Santa Claus', 'Daniel', 'Adam', 'Bill', 'Jessie', 'Sam', 'Giovanni'])

        if voice_type == 'Female':
            voice_selected = st.sidebar.selectbox("Select the voice type:", [
                'Rachel', 'Domi', 'Sarah', 'Emily', 'Elli', 'Dorothy', 'Charlotte', 
                'Matilda', 'Gigi', 'Freya', 'Grace', 'Lily', 'Serena', 'Nicole', 'Glinda', 'Mimi'
            ])

        num_voice_files = st.sidebar.number_input("Enter the number of voice files:", min_value=1, value=1, step=1)

        if st.sidebar.button("Reset voice clips"):
          st.session_state.uploaded_audio_files = []

        # Pose style
        st.sidebar.subheader("Video Settings")
        pose_style = st.sidebar.slider("Select the pose style", 0 , 45)

        # Preprocess mode
        mode = st.sidebar.selectbox("Select the video type:", ['extcrop', 'crop', 'full', 'extfull', 'resize'])

        # Still mode enable
        still = st.sidebar.toggle('Enable still mode', help="This will reduce the movements of the head")
        enhance = st.sidebar.toggle("Enable Face Enhancer", help="This will enhace the face. Specially mouth area. Note that this will increase the time taken to generate videos.")
        
        audio_or_text = st.selectbox("Choose an option", ['Text input', 'Audio file'])
        
        image_col, audio_col = st.columns(2)

        with image_col:
            st.subheader(":red[Upload a portrait image HERE:]")
            uploaded_image = st.file_uploader("Click browse files to upload", type=["jpg", "jpeg", "png"])
            if uploaded_image:
                st.image(uploaded_image, width=300)

        with audio_col:
            if audio_or_text == "Text input":
                st.subheader(":red[Type the text HERE:]")
                # Create text inputs for each voice file
                text_list = []
                for i in range(num_voice_files):
                    text_input = st.text_area(f"Enter Text for Voice File {i+1}")
                    text_list.append(text_input)

            elif audio_or_text == "Audio file":
                st.subheader(":red[Upload the audio files HERE:]")
                uploaded_audio_files = st.file_uploader("Upload Audio Files HERE", type=["mp3"], accept_multiple_files=True)
            
        results_directory = "./results"
        #st.session_state.uploaded_audio_files = []

        if st.button("Create voice clips"):
            if (len(st.session_state.uploaded_audio_files) != 0):
                st.warning("Please reset voice clips first.")
            else:
                if audio_or_text == "Text input":
                    with st.status('Generating audio files....', expanded=True):
                        for i, text_input in enumerate(text_list):
                            voice_id = voice_dict[voice_selected]
                            voice = text_to_speech(text_input, voice_id)
                            mp3_file_path = os.path.join(results_directory, f"input_audio_{i}.mp3")
                            #st.write(mp3_file_path)
                            st.session_state.uploaded_audio_files.append(mp3_file_path)
                            audio_segment = AudioSegment.from_file(BytesIO(voice), format="mp3")
                            audio_segment.export(mp3_file_path, format="mp3")
                            st.audio(voice, format="audio/mp3")

                elif audio_or_text == "Audio file":
                    with st.status('Generating audio files....', expanded=True):
                        for i, voice_audio in enumerate(uploaded_audio_files):
                            mp3_file_path = os.path.join(results_directory, f"input_audio_{i}.mp3")
                            with open(mp3_file_path, "wb") as f:
                                f.write(voice_audio.getbuffer())
                            #st.write(mp3_file_path)
                            st.session_state.uploaded_audio_files.append(mp3_file_path)
                            # audio_segment = AudioSegment.from_file(voice_audio, format="mp3")
                            # audio_segment.export(mp3_file_path, format="mp3")
                            st.audio(voice_audio, format="audio/mp3")
        # with audio_col:
        #     st.subheader(":red[Type the text HERE:]")
        #     # Create text inputs for each voice file
        #     text_list = []
        #     for i in range(num_voice_files):
        #         text_input = st.text_area(f"Enter Text for Voice File {i+1}")
        #         text_list.append(text_input)
            
        # results_directory = "/content/drive/MyDrive/Streamlit/results"
        # #st.session_state.uploaded_audio_files = []

        # if st.button("Create voice clips"):
        #   if (len(st.session_state.uploaded_audio_files) != 0):
        #     st.warning("Please reset voice clips first.")
        #   else:
        #     with st.status('Generating audio files....', expanded=True):
        #       for i, text_input in enumerate(text_list):
        #           voice_id = voice_dict[voice_selected]
        #           voice = text_to_speech(text_input, voice_id)
        #           mp3_file_path = os.path.join(results_directory, f"input_audio_{i}.mp3")
        #           #st.write(mp3_file_path)
        #           st.session_state.uploaded_audio_files.append(mp3_file_path)
        #           audio_segment = AudioSegment.from_file(BytesIO(voice), format="mp3")
        #           audio_segment.export(mp3_file_path, format="mp3")
        #           st.audio(voice, format="audio/mp3")
            #st.write(len(st.session_state.uploaded_audio_files))

        if st.button("Generate Videos", type="primary"):
            if not (uploaded_image or st.session_state.uploaded_audio_files):
                st.warning("Please upload image and audio files first")
            else:
                with st.status('Generating videos....', expanded=True):
                  start_time = time.time()
                  generate_videos(enhance, mode, still, pose_style, uploaded_image, st.session_state.uploaded_audio_files)
                  end_time = time.time()
                  elapsed_time = end_time - start_time
                  st.write(f"Time taken: {elapsed_time:.6f} seconds")

        #st.video("/content/drive/MyDrive/Streamlit/results/output_0.mp4")
    
    # Play videos page
    if rad == "Play Videos":
        st.sidebar.header("Play Settings")
        if st.session_state.uploaded_audio_files is not None:
            play_video(st.session_state.uploaded_audio_files)
        else:
            st.warning("Please generate videos first.")

    # Speech to text with whisper model and generating answers using llama
    if rad == "Any questions!":
        Method = st.selectbox("Select the method of voice input", ["Voice recording", "Audio file"])

        if Method == "Audio file":
            audio = st.file_uploader("Upload audio file", type=['mp3','wav'])

            if audio is not None:  # Check if a file has been uploaded
                if st.button("Generate the text"):
                    output_whisper = speech2text(audio)
                    #output_aai = voice2text(audio)
                    st.write(output_whisper)
                    #st.write(output_aai)
                    res = llama(output_whisper['text'])
                    st.write(res)
                    #output_audio_path = text_to_speech(res,"29vD33N1CtxCmqQRPOHJ" )
                    if voice_selected is None:
                      st.warning("Select the voice first")
                    else:
                      output_audio_path = text_to_speech(res,voice_selected )
                    audio_path = "output_audio.wav"  # You can choose a different file format and name
                    with open(audio_path, "wb") as audio_file:
                        audio_file.write(output_audio_path)
                    autoplay_audio(audio_path)
            else:
                st.write("Please upload an audio file.")

        if Method == "Voice recording":
            uploaded_image = st.file_uploader("Upload Image HERE", type=["jpg", "jpeg", "png"])
            ask_the_agent(uploaded_image)

        
        #prompt = st.chat_input("Enter the question")

        #if prompt:
        #  res = llama(prompt)
        #  st.write(res)
        #  output_audio_path = text_to_speech(res,"29vD33N1CtxCmqQRPOHJ" )
        #  audio_path = "output_audio.wav"  # You can choose a different file format and name
        #  with open(audio_path, "wb") as audio_file:
        #      audio_file.write(output_audio_path)
        #  autoplay_audio(audio_path)

if __name__ == '__main__':
    main()
