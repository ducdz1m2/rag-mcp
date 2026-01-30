import os
import fitz  # PyMuPDF (Phải cài qua pip install pymupdf)
import pytesseract
from PIL import Image
import io
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# THAY ĐỔI DÒNG NÀY:
from langchain_core.documents import Document

# Không cần cấu hình tesseract_cmd trên Fedora vì nó nằm trong /usr/bin/tesseract

DOCS_PATH = "docs/"
VECTOR_DB_PATH = "faiss_index"

def ocr_image_from_page(page):
    ocr_text = ""
    image_list = page.get_images(full=True)
    
    for img_index, img in enumerate(image_list):
        try:
            xref = img[0]
            base_image = page.parent.extract_image(xref)
            image_bytes = base_image["image"]
            
            image = Image.open(io.BytesIO(image_bytes))
            
            # OCR với cấu hình tối ưu cho tiếng Việt
            text = pytesseract.image_to_string(image, lang='vie+eng')
            
            if text.strip():
                ocr_text += f"\n[Nội dung từ hình ảnh {img_index+1}]:\n{text.strip()}"
        except Exception as e:
            print(f"⚠️ Cảnh báo: Không thể OCR hình ảnh {img_index} trên trang {page.number}: {e}")
            continue
    return ocr_text

def process_pdf_with_ocr(pdf_path):
    doc = fitz.open(pdf_path)
    documents = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # 1. Lấy text thuần
        page_text = page.get_text().strip()
        
        # 2. Lấy text từ ảnh
        image_text = ocr_image_from_page(page)
        
        # Nếu là trang scan hoàn toàn (không có text thuần), ta có thể cân nhắc 
        # dùng thêm tính năng convert toàn bộ trang thành ảnh rồi OCR.
        if not page_text and not image_text:
            # Chuyển trang thành ảnh (DPI=300 để rõ nét)
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            page_text = pytesseract.image_to_string(img, lang='vie+eng')

        combined_content = f"{page_text}\n{image_text}".strip()
        
        if combined_content:
            documents.append(Document(
                page_content=combined_content,
                metadata={"source": os.path.basename(pdf_path), "page": page_num + 1}
            ))
            
    doc.close()
    return documents

def process_pdf_simple(pdf_path):
    """Process PDF without OCR - just extract text directly"""
    doc = fitz.open(pdf_path)
    documents = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Only extract text directly from PDF
        page_text = page.get_text().strip()
        
        # Clean up the text but preserve structure
        if page_text:
            # Remove excessive whitespace but keep some line breaks for structure
            lines = page_text.split('\n')
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if line:  # Skip empty lines
                    cleaned_lines.append(line)
            
            cleaned_text = ' '.join(cleaned_lines)
            
            # Only add if we have meaningful content
            if len(cleaned_text) > 30:  # Lower threshold to catch more content
                documents.append(Document(
                    page_content=cleaned_text,
                    metadata={"source": os.path.basename(pdf_path), "page": page_num + 1}
                ))
            
    doc.close()
    return documents

def build_vector_db():
    print("--- 🚀 PDF Text Extraction with OCR Mode ---")
    
    if not os.path.exists(DOCS_PATH):
        os.makedirs(DOCS_PATH)
        return

    all_docs = []
    pdf_files = [f for f in os.listdir(DOCS_PATH) if f.endswith(".pdf")]
    
    for file in pdf_files:
        print(f"📄 Đang xử lý với OCR: {file}...")
        all_docs.extend(process_pdf_with_ocr(os.path.join(DOCS_PATH, file)))

    if not all_docs:
        print("❌ Không tìm thấy nội dung nào để index!")
        return

    # Increase chunk size for better context
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    texts = splitter.split_documents(all_docs)

    # Use better embedding model for Vietnamese
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(VECTOR_DB_PATH)
    print(f"✅ Đã lưu thành công {len(texts)} chunks!")

if __name__ == "__main__":
    build_vector_db()