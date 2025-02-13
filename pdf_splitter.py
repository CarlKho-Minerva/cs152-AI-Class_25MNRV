import argparse
from PyPDF2 import PdfReader, PdfWriter


def pdf_splitter(input_pdf_path, output_pdf_path, start_page, end_page):
    """
    Splits a PDF file into a new PDF containing only the specified page range.

    Args:
        input_pdf_path (str): Path to the input PDF file.
        output_pdf_path (str): Path to the output PDF file.
        start_page (int): The first page to include (1-based index).
        end_page (int): The last page to include (inclusive, 1-based index).
    """
    try:
        with open(input_pdf_path, "rb") as input_pdf:
            reader = PdfReader(input_pdf)
            writer = PdfWriter()

            # Adjust page indices to be 0-based
            start_page -= 1
            # end_page is inclusive, so no need to adjust for slicing

            for page_num in range(start_page, end_page):
                page = reader.pages[page_num]
                writer.add_page(page)

            with open(output_pdf_path, "wb") as output_pdf:
                writer.write(output_pdf)

        print(
            f"Successfully extracted pages {start_page + 1} to {end_page} and saved to {output_pdf_path}"
        )

    except FileNotFoundError:
        print(f"Error: Input PDF file not found at {input_pdf_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # Call the pdf_splitter function directly with your desired parameters
    input_pdf = "ai.pdf"
    output_pdf = "ai_extracted.pdf"
    start_page = 1
    end_page = 5

    pdf_splitter(input_pdf, output_pdf, start_page, end_page)
