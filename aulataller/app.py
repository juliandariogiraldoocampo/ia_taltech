import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image
from streamlit_image_comparison import image_comparison
import io
import tempfile

st.set_page_config(page_title="Vigilancia", page_icon="✈️", layout="centered")
st.header('✈️ Vigilancia de Pasajeros y Equipajes')

MODELOS = {
    "Nano - más rápido, menos preciso": "yolov8n.pt",
    "Small - Equilibrado": "yolov8s.pt",
    "Medium - más preciso, más lento": "yolov8m.pt",
}

seleccion = st.selectbox("Modelo de detección", list(MODELOS.keys()), index=0)

@st.cache_resource
def cargar_modelo(nombre_archivo):
    return YOLO(nombre_archivo)

archivo = st.file_uploader("Subir imagen", type=["jpg", "jpeg", "png", "webp"])

if archivo:
    imagen = Image.open(archivo).convert("RGB")
    modelo = cargar_modelo(MODELOS[seleccion])

    CLASES_COCO = [0, 24, 26, 28] # personas, mochilas, maletas, equipaje de mano

    with st.spinner("Procesando imagen..."):
        resultado = modelo(np.array(imagen), conf=0.25, classes=CLASES_COCO, verbose=False)[0]
    
    personas = 0
    equipajes = 0

    for box in resultado.boxes:
        clase = resultado.names[int(box.cls[0])]
        if clase == "person":
            personas += 1
        elif clase in ['suitcase', 'backpack', 'handbag']:
            equipajes += 1

    col1, col2 = st.columns(2)
    col1.metric("Personas detectadas", personas)
    col2.metric("Equipajes detectados", equipajes)

    imagen_anotada = Image.fromarray(resultado.plot())
    image_comparison(img1=imagen, img2=imagen_anotada, label1="Original", label2="Detección") # type: ignore

    buf = io.BytesIO()
    imagen_anotada.save(buf, format="PNG")
    st.download_button(
        label="Descargar imagen anotada",
        data=buf.getvalue(),
        file_name="Deteccion.png",
        mime="image/png"
    )
