import google.generativeai as genai
import os

class YogiLLM:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def generate_code_response(self, mode, query, code_snippet, error_msg, context_docs):
        """
        Builds the prompt for the Yogi Coding Assistant.
        """
        # 1. Build Context String
        rag_context = ""
        if context_docs:
            rag_context = "\n\nRELEVANT KNOWLEDGE BASE:\n" + "\n".join(
                [f"- From {d['source']}: {d['content'][:500]}..." for d in context_docs]
            )

        # 2. Build System Prompt
        prompt = f"""
        You are 'Yogi', a senior Coding Architect and Systems Engineer.
        Your goal is to provide production-ready, efficient, and clean code solutions.
        
        MODE: {mode}
        
        USER QUERY: {query}
        
        USER CODE:
        ```
        {code_snippet if code_snippet else "N/A"}
        ```
        
        ERROR MESSAGE:
        {error_msg if error_msg else "N/A"}
        
        {rag_context}
        
        INSTRUCTIONS:
        1. Be direct and systematic.
        2. If the user's code is inefficient, explain why and optimize it.
        3. Use Markdown for formatting.
        4. Output the final corrected code in a clear code block.
        """
        
        response = self.model.generate_content(prompt)
        return response.text

    def summarize_article(self, text):
        """
        Builds the prompt for the Summarizer.
        """
        prompt = f"""
        You are an expert Technical Summarizer.
        Analyze the following text and provide:
        1. A bulleted list of key insights.
        2. A 'TL;DR' (Too Long; Didn't Read) summary at the very end.
        3. Keep formatting clean and professional.

        TEXT TO SUMMARIZE:
        {text}
        """
        response = self.model.generate_content(prompt)
        return response.text