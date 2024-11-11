# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LLM Chains for executing Retrival Augmented Generation."""
import logging
import os
from pathlib import Path
from typing import Any, Dict, Generator, List

from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from llama_index.core.node_parser import LangchainNodeParser
from llama_index.core.prompts.base import Prompt
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.readers.file import PDFReader, UnstructuredReader
from llama_index.core.base.response.schema import StreamingResponse

from RAG.src.chain_server.base import BaseExample
from RAG.src.chain_server.tracing import langchain_instrumentation_class_wrapper, llama_index_cb_handler
from RAG.src.chain_server.utils import (
    LimitRetrievedNodesLength,
    del_docs_vectorstore_llamaindex,
    get_config,
    get_doc_retriever,
    get_docs_vectorstore_llamaindex,
    get_embedding_model,
    get_llm,
    get_prompts,
    get_text_splitter,
    get_vector_index,
    set_service_context,
)

# prestage the embedding model
embedding_model = get_embedding_model()  # Use a descriptive variable name
set_service_context()
prompts = get_prompts()

logger = logging.getLogger(__name__)
text_splitter = None


# Additional function to handle follow-up LangChain call
def generate_nvidia_context_response(query: str, llm, prompts) -> str:
    """Generate a follow-up response explaining how NVIDIA generative AI services and agent tooling can be applied."""
    # Retrieve the base prompt template
    base_prompt = prompts.get("nvidia_context_template", "")
    
    # Append the specific instruction to the base prompt
    full_prompt = f"{base_prompt}\n\nUser Query: {query}\n\nWith the context available, describe how NVIDIA generative AI services and agent tooling can be used to implement this request."
    
    # Create the prompt template for the LLM
    nvidia_prompt_template = ChatPromptTemplate.from_messages([
        ("system", full_prompt),
    ])
    
    # Set up the chain with the LLM and output parser
    chain = nvidia_prompt_template | llm | StrOutputParser()
    
    # Invoke the chain with the query as input
    nvidia_response = chain.invoke({"input": query})
    
    # Log the response for debugging purposes
    logger.info(f"NVIDIA context response: {nvidia_response}")
    
    return nvidia_response


@langchain_instrumentation_class_wrapper
class QAChatbot(BaseExample):
    def ingest_docs(self, filepath: str, filename: str) -> None:
        """Ingests documents to the VectorDB.
        It's called when the POST endpoint of `/documents` API is invoked.

        Args:
            filepath (str): The path to the document file.
            filename (str): The name of the document file.

        Raises:
            ValueError: If there's an error during document ingestion or the file format is not supported.
        """
        try:
            # Set callback manager for observability
            Settings.callback_manager = CallbackManager([llama_index_cb_handler])
            logger.info(f"Ingesting {filename} in vectorDB")
            _, ext = os.path.splitext(filename)

            # Load data based on file extension
            if ext.lower() == ".pdf":
                loader = PDFReader()
                documents = loader.load_data(file=Path(filepath))
            else:
                loader = UnstructuredReader()
                documents = loader.load_data(file=Path(filepath), split_documents=False)

            # Add filename as metadata to each document
            for document in documents:
                document.metadata = {"filename": filename, "common_field": "all"}
                # do not generate embedding for filename and page_label
                document.excluded_embed_metadata_keys = ["filename", "page_label"]

            # Get vectorstore instance, vectorstore is selected based on environment variable APP_VECTORSTORE_NAME defaults to milvus
            index = get_vector_index()

            global text_splitter
            # Get text splitter instance, text splitter is selected based on environment variable APP_TEXTSPLITTER_MODELNAME
            # tokenizer dimension of text splitter should be same as embedding model
            if not text_splitter:
                text_splitter = get_text_splitter()

            # Create nodes using existing text splitter
            node_parser = LangchainNodeParser(text_splitter)
            nodes = node_parser.get_nodes_from_documents(documents)

            # Ingest document in vectorstore
            index.insert_nodes(nodes)
            logger.info(f"Document {filename} ingested successfully")
        except Exception as e:
            logger.error(f"Failed to ingest document due to exception {e}")
            raise ValueError(f"Failed to upload document: {e}. Please upload an unstructured text document.")

    def get_documents(self) -> List[str]:
        """Retrieves filenames stored in the vector store.
        It's called when the GET endpoint of `/documents` API is invoked.

        Returns:
            List[str]: List of filenames ingested in vectorstore.
        """
        return get_docs_vectorstore_llamaindex()

    def delete_documents(self, filenames: List[str]) -> bool:
        """Delete documents from the vector index.
        It's called when the DELETE endpoint of `/documents` API is invoked.

        Args:
            filenames (List[str]): List of filenames to be deleted from vectorstore.
        """
        return del_docs_vectorstore_llamaindex(filenames)

    def llm_chain(self, query: str, chat_history: List["Message"], **kwargs) -> Generator[str, None, None]:
        """Execute a simple LLM chain with an additional Nvidia context response."""

        logger.info("Using llm to generate response directly without knowledge base.")
        set_service_context(**kwargs)

        # Primary response generation
        prompt = prompts.get("chat_template", "")
        system_message = [("system", prompt)]
        user_input = [("user", "{query_str}")]
        prompt_template = ChatPromptTemplate.from_messages(system_message + user_input)
        llm = get_llm(**kwargs)
        chain = prompt_template | llm | StrOutputParser()
        response_generator = chain.stream({"query_str": query}, config={"callbacks": [self.cb_handler]})

        # Yield response in chunks
        for chunk in response_generator:
            yield chunk

        # Nvidia context response generation
        nvidia_response = generate_nvidia_context_response(query, llm, prompts)

        # Yield Nvidia context response
        yield "\n\nNvidia Context:\n" + nvidia_response

    def rag_chain(self, query: str, chat_history: List["Message"], **kwargs) -> Generator[str, None, None]:
        """Execute a Retrieval Augmented Generation chain with additional Nvidia context response."""

        logger.info("Using RAG to generate response from document")
        set_service_context(**kwargs)

        # Primary RAG response generation
        retriever = get_doc_retriever(num_nodes=get_config().retriever.top_k)
        qa_template = Prompt(prompts.get("rag_template", ""))
        nodes = retriever.retrieve(query)
        if not nodes:
            logger.warning("Retrieval failed to get any relevant context")
            yield "No response generated from LLM; make sure your query is relevant to the ingested document."
            return

        query_engine = RetrieverQueryEngine.from_args(
            retriever, text_qa_template=qa_template, node_postprocessors=[LimitRetrievedNodesLength()], streaming=True,
        )

        try:
            response = query_engine.query(query)
            if isinstance(response, StreamingResponse):
                for chunk in response.response_gen:
                    yield chunk
            else:
                yield response.response
        except Exception as e:
            logger.error(f"Error during response generation: {e}")
            yield f"An error occurred during response generation: {e}"
            return

        # Nvidia context response generation
        llm = get_llm(**kwargs)
        nvidia_response = generate_nvidia_context_response(query, llm, prompts)

        # Yield Nvidia context response
        yield "\n\nNvidia Context:\n" + nvidia_response

    def document_search(self, content: str, num_docs: int) -> List[Dict[str, Any]]:
        """Search for the most relevant documents for the given search parameters.   

        It's called when the `/search` API is invoked.

        Args:
            content (str): Query to be searched from vectorstore.
            num_docs (int): Number of similar docs to be retrieved from vectorstore.
        """

        try:
            # Get retriever instance
            retriever = get_doc_retriever(num_nodes=num_docs)
            nodes = retriever.retrieve(content)
            output = []
            for node in nodes:
                file_name = node.metadata["filename"]  # Get filename from the current node
                entry = {"score": node.score, "source": file_name, "content": node.text}
                output.append(entry)

            return output

        except Exception as e:
            logger.error(f"Error from /documentSearch endpoint. Error details: {e}")
            return []