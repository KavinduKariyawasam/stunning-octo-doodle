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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path
from generate_videos import generate_videos
from voices import voice_dict
import multiprocessing
from streamlit.runtime.scriptrunner import add_script_run_ctx

multiprocessing.set_start_method('spawn', True)


def download_model():
    REPO_ID = 'vinthony/SadTalker-V002rc'
    snapshot_download(repo_id=REPO_ID, local_dir='./checkpoints', local_dir_use_symlinks=True)

# Function to play the video and record audio
def play_video():
    uploaded_audio_files = st.session_state.uploaded_audio_files
    result_dir = "/content/drive/MyDrive/Streamlit/results"
    st.markdown(
        f'<style>video {{ width: 100%; max-width: 300px; height: 300px; }}</style>',
        unsafe_allow_html=True
    )

    if st.sidebar.button("Play Previous Video") and st.session_state.video_index > 0:
        st.session_state.video_index -= 1

    if st.sidebar.button("Play Next Video") and st.session_state.video_index < len(uploaded_audio_files) - 1:
        st.session_state.video_index += 1
        #st.write(st.session_state.video_index)
    elif st.session_state.video_index > len(uploaded_audio_files):
        st.balloons()
        st.write(st.session_state.video_index)

    

    if st.session_state.video_index < len(uploaded_audio_files):
        st.markdown(f"Question {st.session_state.video_index + 1}")
        video_path = open(os.path.join(result_dir, f"output_{st.session_state.video_index}.mp4"), 'rb')
        st.video(video_path, format='video/mp4')
        
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
                
                #if audio:     
                    

                    # Store the audio file path in the session state
                #   st.session_state[audio_key] = audio_path
                  #  del audio

                if st.button("Save answer"):
                    audio_filename = f"answer_{st.session_state.video_index}.mp3"
                    audio_path = os.path.join(result_dir, audio_filename)
                    with open(audio_path, 'wb') as audio_file:
                        audio_file.write(audio['bytes'])
                    st.success("Saving process successfully completed..")
              
                if st.button("Play recorded audio"):
                    audio_filename = f"answer_{st.session_state.video_index}.mp3"
                    audio_path = os.path.join(result_dir, audio_filename)
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
            #if text:
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
    
    

# Text to voice with elevenlabs api
def text_to_speech(text, voice_id):
    CHUNK_SIZE = 1024
    url = "https://api.elevenlabs.io/v1/text-to-speech/" + voice_id

    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": "b193bcf8431a47bc0d6d254f75eaaedd"
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

#Speech to text with whisper-large-v3 by openai
def speech2text(filename):
    st.audio(filename)
    API_TOKEN = "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"
    API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    #with open(filename, "rb") as f:
    #    data = f.read()
    audio_content = filename.read()
    response = requests.post(API_URL, headers=headers, data=audio_content)
    return response.json()

#LLM llama v2 model for generating answers
def llama(prompt):
    os.environ["REPLICATE_API_TOKEN"] = "r8_CD5rbfDS9Tql0Nh9aK94n8KCoIpWMaP3WHrCJ"
    api = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"])
    output = api.run(
        "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
            input={"prompt": prompt}
        )
    out = ''
    for item in output:
        out += item 
        #print(item, end="")
    return out


# Function to autoplay audio files
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

def process_text(i, text_input, voice_id, results_directory):
    voice = text_to_speech(text_input, voice_id)
    mp3_file_path = os.path.join(results_directory, f"input_audio_{i}.mp3")
    audio_segment = AudioSegment.from_file(BytesIO(voice), format="mp3")
    audio_segment.export(mp3_file_path, format="mp3")
    return mp3_file_path, voice



#@jit(nopython=True)
def process_video(index, audio_path, image_path, pose_style, still, mode, enhancer):
    # Define parameters here
    # Update parameters for this process if needed
    result_dir = './results'
    size = 256
    expression_scale = 1.
    checkpoint_dir = './checkpoints'
    preprocess = 'extcrop'
    verbose = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 2
    input_yaw_list = None
    input_pitch_list = None
    input_roll_list = None
    expression_scale = 1.
    cpu = False
    still = still

    # torch.backends.cudnn.enabled = False

    save_dir = os.path.join(result_dir, strftime(f"output_{index}"))
    os.makedirs(save_dir, exist_ok=True)

    current_root_path = os.path.split(sys.argv[0])[0]

    sadtalker_paths = init_path(checkpoint_dir, os.path.join(current_root_path, 'src/config'), size, preprocess)

    # init model
    preprocess_model = CropAndExtract(sadtalker_paths, device)

    audio_to_coeff = Audio2Coeff(sadtalker_paths,  device)

    animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)

    # crop image and extract 3dmm from image
    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)

    first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(image_path, first_frame_dir, preprocess, source_image_flag=True, pic_size=size)
    if first_coeff_path is None:
        st.write("Can't get the coeffs of the input")
        return

    # audio2ceoff
    batch = get_data(first_coeff_path, audio_path, device, still=still)
    coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style)

    # coeff2video
    data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path,
                                batch_size, input_yaw_list, input_pitch_list, input_roll_list,
                                expression_scale=expression_scale, still_mode=still, preprocess=preprocess, size=size)

    result = animate_from_coeff.generate(data, save_dir, image_path, crop_info, \
                                preprocess=preprocess, img_size=size)

    shutil.move(result, save_dir+ ".mp4")
    st.success(f"Video {index + 1} generated successfully!")

    if not verbose:
        shutil.rmtree(save_dir)






def parallel_video_generation(audio_files, image_path, pose_style, still, mode, enhance):
    with ThreadPoolExecutor() as executor:
        futures = []
        for index, audio_path in enumerate(audio_files):
            future = executor.submit(process_video, index, audio_path, image_path, pose_style, still, mode, enhance)
            futures.append(future)

        for t in executor._threads:
            add_script_run_ctx(t)

        # for future in futures:
        #     future.result()







# Main function for the Streamlit app
def main():
    st.set_page_config(
        page_title='Avatar Creation DEMO',
        page_icon='😎',
        layout='wide',
        initial_sidebar_state='expanded'
    )

    st.title("Avatar Creation DEMO")

    #download_model()

    # Create or get SessionState
    if 'uploaded_audio_files' not in st.session_state:
        st.session_state.uploaded_audio_files = []
        st.session_state.video_index = 0
    
    # Page navigation 
    rad = st.sidebar.radio("Page Navigation", ["Speech to Text","Generate Videos", "Play Videos"])
    
    # Speech to text with whisper model and generating answers using llama
    if rad == "Speech to Text":
      Method = st.selectbox("Select the method of voice input", ["Audio file", "Voice recording"])

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
            output_audio_path = text_to_speech(res,"29vD33N1CtxCmqQRPOHJ" )
            audio_path = "output_audio.wav"  # You can choose a different file format and name
            with open(audio_path, "wb") as audio_file:
                audio_file.write(output_audio_path)
            autoplay_audio(audio_path)
        else:
            st.write("Please upload an audio file.")

      if Method == "Voice recording":
        result_dir = "/content/drive/MyDrive/Streamlit/results"
        audio_bytes = mic_recorder(start_prompt="Start talking", stop_prompt="Stop", key='recorder')
        if audio_bytes:
            #st.audio(audio_bytes['bytes'], format="audio/wav")
            audio_file = io.BytesIO(audio_bytes['bytes'])
            result = speech2text(audio_file)
            st.write("Text result:")
            st.write(result)
      #audio_filename = f"answer.mp3"
      #audio_path = os.path.join("/content/drive/MyDrive/Streamlit/results", audio_filename)
            st.audio(audio_path)
      prompt = st.chat_input("Enter the question")

      if prompt:
        res = llama(prompt)
        st.write(res)
        output_audio_path = text_to_speech(res,"29vD33N1CtxCmqQRPOHJ" )
        audio_path = "output_audio.wav"  # You can choose a different file format and name
        with open(audio_path, "wb") as audio_file:
            audio_file.write(output_audio_path)
        autoplay_audio(audio_path)


    # Generate videos page
    if rad == "Generate Videos":
        st.header("Generate Videos")

        st.subheader("Upload a portrait image HERE:")
        uploaded_image = st.file_uploader("Upload Image HERE", type=["jpg", "jpeg", "png"])

        st.sidebar.header("Audio settings")
        voice_type = st.sidebar.selectbox('Select the gender:', ['Male', 'Female'])

        if voice_type == 'Male':
          voice_selected = st.sidebar.selectbox("Select the voice type:", ['Drew', 'Clyde', 'Paul', 'Dave', 'Fin', 'Antoni', 'Thomas', 'Charlie', 'George', 'Callum', 'Patrick', 'Harry', 'Liam', 'Josh', 'Arnold', 'Matthew', 'James', 'Joseph', 'Jeremy', 'Michael', 'Ethan', 'Santa Claus', 'Daniel', 'Adam', 'Bill', 'Jessie', 'Sam', 'Giovanni'])

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
        mode = st.sidebar.selectbox("Select the video type:", ['crop', 'full', 'extfull', 'extcrop', 'resize'])

        # Still mode enable
        still = st.sidebar.toggle('Enable still mode', help="This will reduce the movements of the head")

        st.subheader("Type the text HERE:")
        # Create text inputs for each voice file
        # text_list = []
        # for i in range(num_voice_files):
        #     text_input = st.text_area(f"Enter Text for Voice File {i+1}")
        #     text_list.append(text_input)
        
        results_directory = "/content/drive/MyDrive/Streamlit/results"
        #st.session_state.uploaded_audio_files = []

        # if st.button("Create voice clips"):
        #     if (len(st.session_state.uploaded_audio_files) != 0):
        #         st.warning("Please reset voice clips first.")
        #     else:
        #         with st.status('Generating audio files....', expanded=True):
        #             with ProcessPoolExecutor() as executor:
        #                 futures = []
        #                 for i, text_input in enumerate(text_list):
        #                     voice_id = voice_dict[voice_selected]
        #                     future = executor.submit(process_text, i, text_input, voice_id, results_directory)
        #                     futures.append(future)

        #                 for future in futures:
        #                     mp3_file_path, voice = future.result()
        #                     st.session_state.uploaded_audio_files.append(mp3_file_path)
        #                     st.audio(voice, format="audio/mp3")
        #                     st.write(st.session_state.uploaded_audio_files)
                    # for i, text_input in enumerate(text_list):
                    #     voice_id = voice_dict[voice_selected]
                    #     voice = text_to_speech(text_input, voice_id)
                    #     mp3_file_path = os.path.join(results_directory, f"input_audio_{i}.mp3")
                    #     #st.write(mp3_file_path)
                    #     st.session_state.uploaded_audio_files.append(mp3_file_path)
                    #     audio_segment = AudioSegment.from_file(BytesIO(voice), format="mp3")
                    #     audio_segment.export(mp3_file_path, format="mp3")
                    #     st.audio(voice, format="audio/mp3")
            #st.write(len(st.session_state.uploaded_audio_files))
        audio_or_text = st.selectbox("Choose an option", ['Text input', 'Audio file'])
        
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

        if st.button("Generate Videos"):
            if not (uploaded_image or st.session_state.uploaded_audio_files):
                st.warning("Please upload image and audio files first")
            else:
                image_path = os.path.join("./results", "input_image.jpg")
                with open(image_path, "wb") as image_file:
                    image_file.write(uploaded_image.getvalue())
                with st.status('Generating videos....', expanded=True):

                    starting_time = time.perf_counter()        


                    parallel_video_generation(st.session_state.uploaded_audio_files, image_path, pose_style, still, mode, None)



                    ending_time = time.perf_counter()

                    print(f"Time taken to generate video(s) is {ending_time - starting_time} second(s)")
                    #generate_videos(None, mode, still, pose_style, uploaded_image, st.session_state.uploaded_audio_files)

        #st.video("/content/drive/MyDrive/Streamlit/results/output_0.mp4")
    
    # Play videos page
    if rad == "Play Videos":
        st.sidebar.header("Play Settings")
        if st.session_state.uploaded_audio_files is not None:
            play_video()
        else:
            st.warning("Please generate videos first.")

if __name__ == '__main__':
    main()