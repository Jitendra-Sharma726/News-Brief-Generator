from groq import Groq

def bullet_point_summary(client, text, num_points = 5) -> str:
    prompt = f"Summarize the following text in {num_points} concise bullet points:\n\n{text}"
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a concise and clear summarizer."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_completion_tokens=300,
    )
    return response.choices[0].message.content.strip()

def abstract_style_summary(client, text, sentence_count = 5) -> str:
    prompt = f"Summarize the following text as a {sentence_count}-sentence abstract:\n\n{text}"
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a concise and clear summarizer."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_completion_tokens=300,
    )
    return response.choices[0].message.content.strip()

def simple_english_summary(client, text, sentence_count = 5) -> str:
    prompt = f"Summarize the following text in simple English suitable for a 12-year-old, in {sentence_count} sentences:\n\n{text}"
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a kind teacher explaining things simply."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_completion_tokens=300,
    )
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    # Ask for API Key
    api_key = input("Enter your Groq API key: ").strip()
    client = Groq(api_key=api_key)

    # Load input article
    filepath = "article.txt"
    with open(filepath, "r") as f:
        content = f.read()

    # Generate summaries
    bullet_summary = bullet_point_summary(client, content, num_points=5)
    abstract_summary = abstract_style_summary(client, content, sentence_count=5)
    simple_summary = simple_english_summary(client, content, sentence_count=5)

    # Print results
    print("\n--- Bullet-point Summary ---\n", bullet_summary)
    print("\n--- Abstract Summary ---\n", abstract_summary)
    print("\n--- Simple English Summary ---\n", simple_summary)
