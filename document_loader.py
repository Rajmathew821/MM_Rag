import base64
import uuid
from pathlib import Path
from unstructured.partition.pdf import partition_pdf

class DocumentLoader:
    def __init__(self):
        pass

    def load_pdf(self, pdf_path):
        try:
            elements = partition_pdf(
                filename=pdf_path,
                strategy='hi_res',
                extract_images_in_pdf=True,
                extract_image_block_types=["Image", "Table"],
                extract_image_block_to_payload=True
            )
            return elements
        except Exception as e:
            raise RuntimeError(f"Error partitioning PDF: {e}")

    def separate_elements(self, elements):
        text_elements, table_elements, image_elements = [], [], []
        Path("output/images").mkdir(parents=True, exist_ok=True)
        Path("output/tables").mkdir(parents=True, exist_ok=True)
        for element in elements:
            doc_id = str(uuid.uuid4())
            element_dict = {"id": doc_id, "metadata": element.metadata.to_dict()}
            if element.category in ['NarrativeText', 'Text']:
                element_dict["content"] = str(element)
                text_elements.append(element_dict)
            elif element.category == 'Table':
                if element.metadata.image_base64:
                    img_file = Path("output/tables") / f"{doc_id}.png"
                    img_file.write_bytes(base64.b64decode(element.metadata.image_base64.replace('\n', '')))
                    element_dict["image_path"] = str(img_file)
                table_elements.append(element_dict)
            elif element.category == 'Image' and element.metadata.image_base64:
                img_file = Path("output/images") / f"{doc_id}.png"
                img_file.write_bytes(base64.b64decode(element.metadata.image_base64.replace('\n', '')))
                element_dict["image_path"] = str(img_file)
                image_elements.append(element_dict)
        return text_elements, table_elements, image_elements
