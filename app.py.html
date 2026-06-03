import streamlit as st
import cv2, numpy as np, io
from PIL import Image
import img2pdf

st.set_page_config(page_title="Scan to PDF", layout="centered")
st.title("📄 Scan Foto jadi PDF A4")
st.write("Upload foto dokumen, auto crop + lurusin, langsung jadi PDF")

def scan_document(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blur, 75, 200)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break
    if screenCnt is not None:
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
        warp = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
        return warp
    return img

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
        pil_img.save(buf, format="JPEG", quality=60, optimize=True)
        gambar_temp.append(buf.getvalue())
        st.image(pil_img, caption=file.name, use_container_width=True)

    if st.button("Buat PDF"):
        A4 = img2pdf.mm_to_pt(210, 297)
        layout = img2pdf.get_layout_fun(A4)
        pdf_bytes = img2pdf.convert(gambar_temp, layout_fun=layout)
        st.download_button("Download PDF", pdf_bytes, "scan.pdf", "application/pdf")