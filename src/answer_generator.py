from openai import OpenAI

def generate_answer(client, query, context_chunks):
    formatted_context = []
    for chunk in context_chunks:
        if chunk.type == "text":
            formatted_context.append(
                f"Page {chunk.page} ({chunk.source}): {chunk.content[:300]}..."
            )
        else:
            formatted_context.append(
                f"Image from page {chunk.page} ({chunk.source})"
            )

    prompt = f"""Analyze the query using both text and visual context. Follow these rules:
    1. Base response strictly on the context
    2. Acknowledge visual elements when relevant
    3. Cite sources using [Page X]
    4. State limitations when uncertain

    Context:
    {" | ".join(formatted_context)}

    Query: {query}

    Answer:"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a scientific paper analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=500
    )
    return response.choices[0].message.content
