import PyPDF2

def extract_text_from_pdf(pdf_path, txt_path):
    """Extract text from a PDF and save it to a text file."""
    with open(pdf_path, "rb") as pdf_file, open(txt_path, "w") as txt_file:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            txt_file.write(page.extract_text())
            txt_file.write("\n")

extract_text_from_pdf("rules.pdf", "rules.txt")
extract_text_from_pdf("principles.pdf", "principles.txt")

print("Text extracted and saved to rules.txt and principles.txt")
