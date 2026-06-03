import streamlit as st
import cv2
import numpy as np
import io
from PIL import Image
import img2pdf

st.set_page_config(page_title="Scan to PDF", layout="centered")
st.title("📄 Scan Foto jadi PDF A4")
st.write("Upload foto dokumen, auto crop + lurusin, langsung jadi PDF")

def scan_document(img):
    h, w = img.shape[:2]
    orig = img.copy()

    # Resize biar proses lebih cepat + stabil
    ratio = 800.0 / max(h, w)
    img_resized = cv2.resize(img, (int(w*ratio), int(h*ratio)))

    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    # Pakai adaptive threshold biar tahan sama bayangan
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Cari kontur segi-4 terbesar yang mirip rasio A4
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
        # Scale balik ke ukuran asli
        screenCnt = screenCnt / ratio