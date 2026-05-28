# src/chatbot.py
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

def generate_final_response(user_query: str, unified_context: str, api_key: str) -> str:
    """
    Modular generation stage compliant with AGAI-03 Phase 4 guidelines.
    Constructs anti-hallucination boundaries and invokes Gemini.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        google_api_key=api_key
    )
    
    system_prompt = (
        "You are an expert academic advisor representing the KU Leuven International Admissions Office.\n"
        "Your absolute objective is to resolve student inquiries using ONLY the verified context blocks provided below.\n\n"
        f"[VERIFIED CONTEXT BLOCKS]:\n{unified_context}\n\n"
        "Strict System Guardrails:\n"
        "1. Answer the student's question accurately, maintaining a formal and helpful tone.\n"
        "2. Base your response purely on the context above. Do NOT assume, generalize, or extrapolate.\n"
        "3. Crucially, if the context does not contain the specific facts required to formulate an answer, you MUST reply verbatim with:\n"
        "   'I am sorry, but I cannot find that specific detail in the current crawled pages.'\n\n"
        f"Student Inquiry: {user_query}"
    )
    
    prompt_template = ChatPromptTemplate.from_messages([("user", system_prompt)])
    chain = prompt_template | llm
    response = chain.invoke({})
    
    # Secure sequence normalization
    raw_content = response.content
    if isinstance(raw_content, list):
        parsed_chunks = []
        for segment in raw_content:
            if isinstance(segment, str): parsed_chunks.append(segment)
            elif isinstance(segment, dict) and "text" in segment: parsed_chunks.append(segment["text"])
        return "".join(parsed_chunks).strip()
    
    return str(raw_content).strip()