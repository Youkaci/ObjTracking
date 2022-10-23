# To run use
# $ streamlit run yolor_streamlit_demo.py

from yolo_v7 import names, load_yolov7_and_process_each_frame

import tempfile
import cv2


from models.models import *
from utils.datasets import *
from utils.general import *

import streamlit as st


def main():
    st.title("Hydranet (Segmentation and Depth) and Object Tracking")

    st.sidebar.title("Operations")

    st.markdown( """
 
    <style>
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        background-color: black;
    }
    body {background-color: powderblue;}

    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
                 )


    use_webcam = st.sidebar.checkbox('Enable Webcam')


    st.sidebar.markdown('---')
    confidence = st.sidebar.slider('Confidence Bar', min_value=0.0, max_value=1.0, value=0.25)
    st.sidebar.markdown('---')

    save_img = st.sidebar.checkbox('Save Video')
    enable_GPU = st.sidebar.checkbox('enable GPU')
    segm = st.sidebar.checkbox('Segm')
    depth = st.sidebar.checkbox('depth')

    custom_classes = st.sidebar.checkbox('Use Custom Classes')
    assigned_class_id = []
    if custom_classes:
        assigned_class = st.sidebar.multiselect('Choose Classes for Obj Track', list(names), default='car')
        for each in assigned_class:
            assigned_class_id.append(names.index(each))

    video_file_buffer = st.sidebar.file_uploader("Video Formats", type=["mp4", "mov", 'avi', 'asf', 'm4v'])

    DEMO_VIDEO = 'test.mp4'

    tfflie = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)

    ##We get our input video here

    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0, cv2.CAP_ARAVIS)
            tfflie.name = 0
        else:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tfflie.name = DEMO_VIDEO
            dem_vid = open(tfflie.name, 'rb')
            demo_bytes = dem_vid.read()

            st.sidebar.text('Original Video')
            st.sidebar.video(demo_bytes)

    else:
        tfflie.write(video_file_buffer.read())
        # print("No Buffer")
        dem_vid = open(tfflie.name, 'rb')
        demo_bytes = dem_vid.read()

        st.sidebar.text('Original Video')
        st.sidebar.video(demo_bytes)

    print(tfflie.name)
    # vid = cv2.VideoCapture(tfflie.name)

    stframe = st.empty()

    st.markdown("<hr/>", unsafe_allow_html=True)
    kpi1, kpi2, kpi3 = st.columns(3)  # st.columns(3)

    # stframe.image(im0,channels = 'BGR',use_column_width=True)

    with kpi1:
        st.markdown("**Frame Rate**")
        kpi1_text = st.markdown("0")

    with kpi2:
        st.markdown("**Tracked Objects**")
        kpi2_text = st.markdown("0")

    with kpi3:
        st.markdown("**Total Count**")
        kpi3_text = st.markdown("0")

    st.markdown("<hr/>", unsafe_allow_html=True)
    # call yolor
    # load_yolor_and_process_each_frame(tfflie.name, enable_GPU, confidence, assigned_class_id, kpi1_text, kpi2_text, kpi3_text, stframe)
    load_yolov7_and_process_each_frame('yolov7', tfflie.name, enable_GPU, save_img, confidence, assigned_class_id,
                                       kpi1_text, kpi2_text, kpi3_text, stframe)

    st.text('Video is Processed')


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass

