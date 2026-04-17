import csv
from pathlib import Path
from llama_index.core import Document

def load_documents(csv_path: str = "./DATA/pyq_questions.csv") -> list[Document]:
    """Load questions from CSV and return a list of LlamaIndex Documents."""
    documents = []
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")
    with open(csv_path, newline="", encoding="latin-1") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        print(f"CSV Headers: {headers}")
        for row in reader:
            # Strip whitespace from all keys
            row = {k.strip(): v.strip() for k, v in row.items()}
            # Skip rows where 'year' is not a valid number
            if not row.get("year", "").isdigit():
                continue
            # Build the image path only if a filename exists
            q_image = row.get("Q_Image", "").strip()
            image_path = f"./DATA/Q_Images/{{q_image}}" if q_image else ""
            doc = Document(
                text=row["question"],
                metadata={
                    "topics": row.get("topics", ""),
                    "answer": row.get("Ans", ""),
                    "answer_description": row.get("Ans_des", ""),
                    "exam": row.get("exam", ""),
                    "year": int(row["year"]),
                    "subject": row.get("subject", ""),
                    "image": image_path,
                    # Store question in metadata too for easy access
                    "question": row["question"],
                },
            )
            documents.append(doc)
    print(f"Loaded {{len(documents)}} documents")
    return documents
