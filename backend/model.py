from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("API_KEY")

client = Groq(api_key=api_key)


def generate_answer(query, top_chunks,similarity_score):

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",

        messages=[
            {
                "role": "system",
                "content": (
                    f"""
                    You are an AI assistant.
                    Answer ONLY using the given context.
                    If query is out of context or the scores are low say "This is not in your source" and then add the out of context details something that starts like "This is not from your source, but what I can give is (what you want to say)"

                    Context:
                    {top_chunks}

                    Scores:
                    {similarity_score}
                    """
                )
            },
            {
                "role": "user",
                "content": query
            }
        ],

        temperature=0.2,
        max_tokens=200
    )

    response = completion.choices[0].message.content

    return response