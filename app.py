import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from fpdf import FPDF

st.set_page_config(page_title="Scan to PDF", layout="centered")
st.title("📄 Scan Foto jadi PDF A4")
st.write("Upload foto dokumen, auto crop + lurusin, langsung jadi PDF")

def scan_document(img):
    orig = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blur, 75, 200)
    
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    
    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break
    
    if screenCnt is None:
        return img
    
    # Lurusin rotasi dulu
    rect = cv2.minAreaRect(screenCnt)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    # Baru crop perspektif
    gray_rot = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    blur_rot = cv2.GaussianBlur(gray_rot, (5,5), 0)
    edged_rot = cv2.Canny(blur_rot, 75, 200)
    cnts_rot, _ = cv2.findContours(edged_rot.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts_rot = sorted(cnts_rot, key=cv2.contourArea, reverse=True)[:5]
    
    for c in cnts_rot:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break
    
    warped = four_point_transform(rotated, screenCnt.reshape(4, 2))
    return warped
    return orig

uploaded_files = st.file_uploader("Upload foto dokumen, bisa banyak sekaligus",
                                   type=['jpg','jpeg','png'],
                                   accept_multiple_files=True)

if uploaded_files:
    gambar_temp = []
    for file in uploaded_files:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img = scan_document(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=70)
        gambar_temp.append(buf)
        st.image(pil_img, caption=file.name, use_container_width=True)

    if st.button("Buat PDF"):
        pdf = FPDF(unit="pt")
        for img_buf in gambar_temp:
            img_buf.seek(0)
            img = Image.open(img_buf)
            w, h = img.size
            pdf.add_page(format=(w, h))
            pdf.image(img_buf, x=0, y=0, w=w, h=h)
        pdf_output = bytes(pdf.output())
        st.download_button("Download PDF", pdf_output, "scan.pdf", "application/pdf")