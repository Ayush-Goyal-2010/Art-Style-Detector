from ultralytics import YOLO
import streamlit as st

from PIL import Image

st.set_page_config(layout = "wide")

def models():
    mod = YOLO('runs\classify\\train\weights\\best.pt')
    return mod

st.title("Art Style Detector")

tab1, tab2 = st.tabs(["About Art Styles", "Detect Art Styles"])

with tab1:
    st.write("There are majorly 5 art styles in the huge realm of art")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("1. Drawings")
        st.image("assets\\drawing.jpg")
        st.write("Drawings are one of the oldest and simplest forms of expression, used to communicate ideas, tell stories, and capture the world around us. With just lines, shapes, and shading, drawings can convey emotions, imagination, and detail that words sometimes cannot. They serve as the foundation of art, design, and even technical fields like engineering and architecture, where sketches help visualize concepts before they take final form. Whether simple doodles or intricate masterpieces, drawings reflect creativity and the unique perspective of the artist.")
    with col2:
        st.header("2. Engravings")
        st.image("assets\\engraving.jpg")
        st.write("Engravings are designs or images carved onto hard surfaces such as metal, wood, or stone, often using sharp tools or modern techniques like lasers. They have been used for centuries to decorate objects, record important information, and create detailed prints in art and publishing. Because of their precision and durability, engravings are valued both for their beauty and their ability to preserve messages or images over time.")
    with col3:
        st.header("3. Iconography")
        st.image("assets\\iconography.jpg", use_container_width = True)
        st.write("Iconography is the study and use of symbols and images to represent ideas, beliefs, or concepts. It is commonly seen in art, religion, and culture, where specific icons or motifs carry deeper meanings beyond their visual form. By understanding iconography, we can interpret the hidden stories, traditions, and values that different societies express through their imagery.")
    col4, col5 = st.columns(2)
    with col4:
        st.header("4. Painting")
        st.image("assets\\painting.jpg")
        st.write("Paintings are a form of visual art where colors, shapes, and textures are applied to a surface such as canvas, paper, or walls to create expressive images. They can capture reality, tell stories, or convey emotions and imagination in ways words cannot. Throughout history, paintings have been used for decoration, religious expression, documentation, and pure creativity, making them one of the most powerful and enduring mediums of human expression.")
    with col5:
        st.header("5. Sculpture")
        st.image("assets\\sculpture.jpg")
        st.write("Sculpture is a three-dimensional form of art created by shaping or carving materials like stone, wood, clay, or metal. Unlike paintings or drawings, sculptures occupy real space and can be viewed from different angles, giving them a sense of depth and presence. They have been used throughout history to honor gods, leaders, and cultural traditions, while also serving as a medium for artists to express ideas and emotions in a tangible, lasting form.")

    

with tab2:
    img = st.file_uploader('Upload Your Image', type = ['jpg', 'png', 'jpeg'])
    analyse = st.button("Analyse/Submit")

    if analyse:
        if img is not None:
            img = Image.open(img)
            st.markdown('Image Visualisation')
            st.image(img)
            st.subheader("Art Style")
            model = models()
            res = model.predict(img)
            label = res[0].probs.top5
            conf = res[0].probs.top5conf
            conf = conf.tolist()
            st.write("Art Style: " + str(res[0].names[label[0]].title()))
            st.write("Confidence Level: " + str(conf[0]))