from groq import Groq

def bullet_point_summary(client, text, num_points=5) -> str:
    """
    Summarize text into concise bullet points.
    """
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


def abstract_style_summary(client, text, sentence_count=5) -> str:
    """
    Summarize text as a brief abstract.
    """
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


def simple_english_summary(client, text, sentence_count=5) -> str:
    """
    Summarize text in simple English for a 12-year-old reader.
    """
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


# Keyword extractor function
def extract_keywords(text):
    """
    Extract keywords: lowercase, strip punctuation, ignore short words.
    """
    return {
        word.lower().strip(".,!?")
        for word in text.split()
        if len(word) > 4
    }


# Choose best summary (Keyword Overlap)
def best_summary_by_keywords(article, summaries) -> str:
    """
    Choose the best summary by measuring keyword overlap with article.
    """
    article_keywords = extract_keywords(article)
    best_label, best_summary, best_score = None, None, -1

    for label, summary in summaries.items():
        summary_keywords = extract_keywords(summary)
        overlap = len(article_keywords.intersection(summary_keywords))
        score = overlap / (len(article_keywords) + 1)

        print(f"Keyword overlap score for {label}: {score:.4f}")

        if score > best_score:
            best_label, best_summary, best_score = label, summary, score

    return f"Best Summary (by keywords: {best_label}):\n{best_summary}"


if __name__ == "__main__":
    api_key = input("Enter your Groq API key: ").strip()
    client = Groq(api_key=api_key)

    filepath = "article.txt"
    with open(filepath, "r") as f:
        content = f.read()

    bullet_summary = bullet_point_summary(client, content, num_points=5)
    abstract_summary = abstract_style_summary(client, content, sentence_count=5)
    simple_summary = simple_english_summary(client, content, sentence_count=5)

    print("\n--- Bullet-point Summary ---\n", bullet_summary)
    print("\n--- Abstract Summary ---\n", abstract_summary)
    print("\n--- Simple English Summary ---\n", simple_summary)

    summaries = {
        "Bullet Points": bullet_summary,
        "Abstract": abstract_summary,
        "Simple English": simple_summary,
    }

    final_summary = best_summary_by_keywords(content, summaries)
    print("\nFinal Chosen Summary:\n", final_summary)
