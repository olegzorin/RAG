import re
import camelot
import spacy
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from spacy.cli import download
from tqdm import tqdm

# example_file_path = 'file:/Users/oleg/Downloads/AetnaExample.pdf'
example_file_path = 'https://s3.amazonaws.com/sbox.ragdoc/99?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240807T121438Z&X-Amz-SignedHeaders=host&X-Amz-Credential=AKIAJPAZDO6JTXZ6FQSQ%2F20240807%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Expires=3600&X-Amz-Signature=f77906ece6a20391fbecb4038599271bb38989bc047fd59027ed0be342609556'


def load_pdf(file_path, chunk_size, chunk_overlap):
    # load pdf with Langchain loader

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # get total pages count
    page_count = len(documents)

    # text cleaning and normalization
    for document in tqdm(documents):
        # text cleaning
        document.page_content = normalize(document.page_content)
        print(document.page_content)

        # add metadata 'text' classification
        document.metadata["page"] = str(document.metadata["page"] + 1)
        document.metadata["type"] = "text"

    # create text chunks base on character splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text_chunks = text_splitter.split_documents(documents)

    return text_chunks, page_count


def get_tables(path: str, pages: int):
    tables_result, metadata_result, tables_summary = [], [], []

    for page in tqdm(range(1, pages)):

        table_list = camelot.read_pdf(path, pages=str(page))

        for tab in range(table_list.n):

            df_table = table_list[tab].df.dropna(how="all").loc[:, ~table_list[tab].df.columns.isin(['', ' '])]
            df_table = df_table.apply(lambda x: x.str.replace("\n", " "))

            df_table = df_table.rename(columns=df_table.iloc[0]).drop(df_table.index[0]).reset_index(drop=True)

            if df_table.shape[0] <= 3 or df_table.eq("").all(axis=None):
                continue

            metadata_table = {"source": path, "page": str(page), "type": "row"}

            df_table["summary"] = df_table.apply(
                lambda x: re.sub('[\0\r\n]+', ' ', " ".join([f"{col}: {val}, " for col, val in x.items()])), #.replace("\xa0", " ").replace("\x00", ' '),
                axis=1
            )

            docs_summary = [Document(page_content=row["summary"].strip(), metadata=metadata_table) for _, row in df_table.iterrows()]

            tables_result.append(df_table)
            metadata_result.append(metadata_table)
            tables_summary.extend(docs_summary)

            metadata_table = {"source": path, "page": str(page), "type": "table"}
            # tables_summary.append(Document(page_content=df_table.to_markdown(), metadata=metadata_table))
            metadata_result.append(metadata_table)

    return tables_summary, metadata_result, tables_result


def get_all_tables(path: str):
    tables_summary = []
    table_list = camelot.read_pdf(path, pages="all")
    for tab in range(table_list.n):
        df_table = table_list[tab].df.dropna(how="all").loc[:, ~table_list[tab].df.columns.isin(['', ' '])]
        df_table = df_table.apply(lambda x: x.str.replace("\n", " "))
        df_table = df_table.rename(columns=df_table.iloc[0]).drop(df_table.index[0]).reset_index(drop=True)
        if df_table.shape[0] <= 3 or df_table.eq("").all(axis=None):
            continue
        df_table["summary"] = df_table.apply(
            lambda x: re.sub('[\0\r\n\\s]+', ' ', " ".join([f"{col}: {val}, " for col, val in x.items()])),
            axis=1
        )
        tables_summary.extend(row["summary"].strip() for _, row in df_table.iterrows())
    return tables_summary


try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Model not found. Downloading the model...")
    download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')


def normalize(sentence: str) -> str:
    """Normalize the list of sentences and return the same list of normalized sentences"""

    sentence = nlp(re.sub('[\0\r\n\\s]+', ' ', sentence))

    # Convert the sentence to lowercase and process it with spaCy
    # sentence = nlp(sentence.replace('\n', ' ').lower())

    # Lemmatize the words and filter out punctuation, short words, stopwords, mentions, and URLs
    return " ".join([word.lemma_ for word in sentence if (not word.is_punct)
                                    and (len(word.text) > 2) and (not word.is_stop)
                                    and (not word.text.startswith('@')) and (not word.text.startswith('http'))])


# text_chunks, page_count = load_pdf(example_file_path, chunk_size=600, chunk_overlap=100)
# print(f"Total text chunks: {len(text_chunks)}")
# print(f"Total pages: {page_count}")

# tables_summary, metadata_result, tables_result = get_tables(example_file_path, page_count)
tables_summary = get_all_tables(example_file_path)
for x in tables_summary:
    print(x)

