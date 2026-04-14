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







    emojis = {
    "person": "👤", "bicycle": "🚲", "car": "🚗", "motorcycle": "🏍️", "airplane": "✈️",
    "bus": "🚌", "train": "🚂", "truck": "🚚", "boat": "⛵", "traffic light": "🚦",
    "fire hydrant": "🧯", "stop sign": "🛑", "parking meter": "🅿️", "bench": "🪑", "bird": "🐦",
    "cat": "🐱", "dog": "🐶", "horse": "🐴", "sheep": "🐑", "cow": "🐮", "elephant": "🐘",
    "bear": "🐻", "zebra": "🦓", "giraffe": "🦒", "backpack": "🎒", "umbrella": "☔",
    "handbag": "👜", "tie": "👔", "suitcase": "🧳", "frisbee": "🥏", "skis": "⛷️",
    "snowboard": "🏂", "sports ball": "⚽", "kite": "🪁", "baseball bat": "🏏",
    "baseball glove": "🥎", "skateboard": "🛹", "surfboard": "🏄", "tennis racket": "🎾",
    "bottle": "🍾", "wine glass": "🍷", "cup": "☕", "fork": "🍴", "knife": "🔪",
    "spoon": "🥄", "bowl": "🥣", "banana": "🍌", "apple": "🍎", "sandwich": "🥪",
    "orange": "🍊", "broccoli": "🥦", "carrot": "🥕", "hot dog": "🌭", "pizza": "🍕",
    "donut": "🍩", "cake": "🍰", "chair": "🪑", "couch": "🛋️", "potted plant": "🪴",
    "bed": "🛏️", "dining table": "🍽️", "toilet": "🚽", "tv": "📺", "laptop": "💻",
    "mouse": "🖱️", "remote": "📟", "keyboard": "⌨️", "cell phone": "📱", "microwave": "📡",
    "oven": "🔥", "toaster": "🍞", "sink": "🚰", "refrigerator": "🧊", "book": "📖",
    "clock": "⏰", "vase": "🏺", "scissors": "✂️", "teddy bear": "🧸", "hair drier": "💨",
    "toothbrush": "🪥"
}
# Clases organizadas por categoría
categorias = {
    "👤 Personas": [(0, "person")],
    "🚗 Vehículos": [(1, "bicycle"), (2, "car"), (3, "motorcycle"), (4, "airplane"), (5, "bus"), (6, "train"), (7, "truck"), (8, "boat")],
    "🚦 Señales y mobiliario": [(9, "traffic light"), (10, "fire hydrant"), (11, "stop sign"), (12, "parking meter"), (13, "bench")],
    "🐾 Animales": [(14, "bird"), (15, "cat"), (16, "dog"), (17, "horse"), (18, "sheep"), (19, "cow"), (20, "elephant"), (21, "bear"), (22, "zebra"), (23, "giraffe")],
    "🎒 Accesorios": [(24, "backpack"), (25, "umbrella"), (26, "handbag"), (27, "tie"), (28, "suitcase")],
    "🏃 Deportes": [(29, "frisbee"), (30, "skis"), (31, "snowboard"), (32, "sports ball"), (33, "kite"), (34, "baseball bat"), (35, "baseball glove"), (36, "skateboard"), (37, "surfboard"), (38, "tennis racket")],
    "🍽️ Cocina y comida": [(39, "bottle"), (40, "wine glass"), (41, "cup"), (42, "fork"), (43, "knife"), (44, "spoon"), (45, "bowl"), (46, "banana"), (47, "apple"), (48, "sandwich"), (49, "orange"), (50, "broccoli"), (51, "carrot"), (52, "hot dog"), (53, "pizza"), (54, "donut"), (55, "cake")],
    "🛋️ Hogar": [(56, "chair"), (57, "couch"), (58, "potted plant"), (59, "bed"), (60, "dining table"), (61, "toilet"), (62, "tv"), (63, "laptop"), (64, "mouse"), (65, "remote"), (66, "keyboard"), (67, "cell phone")],
    "🔌 Electrodomésticos": [(68, "microwave"), (69, "oven"), (70, "toaster"), (71, "sink"), (72, "refrigerator")],
    "📚 Objetos varios": [(73, "book"), (74, "clock"), (75, "vase"), (76, "scissors"), (77, "teddy bear"), (78, "hair drier"), (79, "toothbrush")]
}


with st.expander("📋 Lista completa de clases COCO (80 objetos)", expanded=False):
    for categoria, clases in categorias.items():
        st.markdown(f"#### {categoria}")
        cols = st.columns(4)
        for idx, (id_clase, nombre) in enumerate(clases):
            col_idx = idx % 4
            emoji = emojis.get(nombre, "🔹")
            with cols[col_idx]:
                st.write(f"**{id_clase}:** {emoji} {nombre}")
    
    st.caption(f"**Total:** 80 clases distribuidas en {len(categorias)} categorías")