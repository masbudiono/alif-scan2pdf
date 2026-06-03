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
    h, w = img.shape[:2]
    orig = img.copy()
    ratio = 800.0 / max(h, w)
    img_resized = cv2.resize(img, (int(w*ratio), int(h*ratio)))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    screenCnt = None
    max_area = 0
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(c)
            x,y,cw,ch = cv2.boundingRect(approx)
            aspect = cw / float(ch)
            if area > max_area and 0.5 < aspect < 2.0 and area > 10000:
                max_area = area
                screenCnt = approx
    if screenCnt is not None:
        screenCnt = screenCnt / ratio
        pts = screenCnt.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0])**2) + ((br[1] - bl[1])**2))
        widthB = np.sqrt(((tr[0] - tl[0])**2) + ((tr[1] - tl[1])**2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0])**2) + ((tr[1] - br[1])**2))
        heightB = np.sqrt(((tl[0] - bl[0])**2) + ((tl[1] - bl[1])**2))
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warp = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
        return warp
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
        pdf = FPDF(unit="mm", format="A4")
        for img_buf in gambar_temp:
            pdf.add_page()
            pdf.image(img_buf, x=10, y=10, w=190)
        pdf_output = bytes(pdf.output())
        st.download_button("Download PDF", pdf_output, "scan.pdf", "application/pdf")