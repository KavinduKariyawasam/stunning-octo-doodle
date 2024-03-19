import streamlit as st
import shutil
import torch
from time import strftime
import os, sys
from pydub import AudioSegment
from io import BytesIO

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path



def generate_videos(enhance, mode, still, pose_style, uploaded_image, uploaded_audio_files):
    st.session_state.video_index = 0

    result_dir = "/content/drive/MyDrive/Streamlit/Results"
    
    if uploaded_image is not None and uploaded_audio_files is not None and len(uploaded_audio_files) > 0:
        image_path = os.path.join(result_dir, "input_image.jpg")
        with open(image_path, "wb") as image_file:
            image_file.write(uploaded_image.getvalue())
        
        # Iterate through audio files and generate videos
        for index, audio_path in enumerate(uploaded_audio_files):
            # Read the content of the audio file into a BytesIO object
            with open(audio_path, "rb") as audio_file:
                audio_content = BytesIO(audio_file.read())

            new_audio_path = os.path.join(result_dir, f"output_audio_{index}.mp3")
            with open(new_audio_path, "wb") as audio_file:
                audio_file.write(audio_content.getvalue())

            # Define parameters here
            pic_path = image_path
            audio_path = audio_path
            result_dir = './results'
            pose_style = pose_style
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            batch_size = 2
            input_yaw_list = None
            input_pitch_list = None
            input_roll_list = None
            checkpoint_dir = './checkpoints'
            size = 256
            expression_scale = 1.2
            head_motion_scale = 0.5
            enhancer = None  # gfpgan or RestoreFormer
            cpu = False
            still = still
            preprocess = mode
            verbose = False

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

            first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(pic_path, first_frame_dir, preprocess, source_image_flag=True, pic_size=size)
            if first_coeff_path is None:
                st.write("Can't get the coeffs of the input")
                return

            # audio2ceoff
            batch = get_data(first_coeff_path, audio_path, device, still=still)
            coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style)

            # coeff2video
            data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path,
                                        batch_size, input_yaw_list, input_pitch_list, input_roll_list,
                                        expression_scale=expression_scale, head_motion_scale=head_motion_scale, still_mode=still, preprocess=preprocess)

            result = animate_from_coeff.generate(data, save_dir, pic_path, crop_info, \
                                         preprocess=preprocess, img_size=size)

            shutil.move(result, save_dir+ ".mp4")
            st.success(f"Video {index + 1} generated successfully!")

            if not verbose:
                shutil.rmtree(save_dir)

            st.session_state.video_index = 0
        st.success("All videos have created successfully.")

    else:
      if uploaded_image is None:
        st.warning("Please upload an portrait image first.")
      else:
        st.write(len(uploaded_audio_files))
        st.warning("Please check the audio files")