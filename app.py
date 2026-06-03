import streamlit as st
import cv2
import numpy as np
from fpdf import FPDF
import io
from PIL import Image

st.set_page_config(page_title="Scan2PDF", layout="wide")
st.title("Scan2PDF - Auto Crop & Auto Lurus")

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

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

uploaded_files = st.file_uploader("Upload foto dokumen", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

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
        st.download_button("Download PDF",