from openai import OpenAI
import google.generativeai as palm
import anthropic

openai_client = OpenAI(api_key='YOUR_OPENAOI_API_KEY')
palm.configure(api_key='YOUR_PALM_API_KEY')
client = anthropic.Anthropic(api_key='ANTHROPIC_API_KEY', )
deepinfra_openai = OpenAI(
    api_key="$DEEPINFRA_TOKEN",
    base_url="https://api.deepinfra.com/v1/openai",
)


# https://platform.openai.com/docs/guides/text-generation/chat-completions-api
def chat_openai(prompt):
    completion = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=1,
        messages=prompt,
    )

    return completion.choices[0].message.content


# https://docs.anthropic.com/en/docs/quickstart-guide
def chat_anthropic(prompt):
    message = client.messages.create(
        model="claude-instant-1.2",
        temperature=1,
        messages=prompt
    )

    print(message.content)


# https://ai.google.dev/palm_docs/text_quickstart
def chat_palm(prompt):
    completion = palm.generate_text(
        model="models/text-bison-002",
        prompt=prompt,
        temperature=1,
    )

    return completion.result


# https://deepinfra.com/meta-llama/Llama-2-70b-chat-hf/api?example=openai-python
def chat_deepinfra(prompt):
    completion = deepinfra_openai.chat.completions.create(
        model="meta-llama/Llama-2-70b-chat-hf",
        messages=prompt,
    )

    return completion.choices[0].message.content


def chat(prompt, llm_type):
    if llm_type == 'openai':
        return chat_openai(prompt)
    elif llm_type == 'anthropic':
        return chat_anthropic(prompt)
    elif llm_type == 'palm':
        return chat_palm(prompt)
    elif llm_type == 'deepinfra':
        return chat_deepinfra(prompt)