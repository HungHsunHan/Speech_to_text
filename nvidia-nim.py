from openai import OpenAI

# client = OpenAI(
#     base_url="https://integrate.api.nvidia.com/v1",
#     api_key="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",
# )

# completion = client.chat.completions.create(
#     model="meta/llama-3.1-405b-instruct",
#     messages=[
#         {
#             "role": "user",
#             "content": "Tell me a joke about Taiwanese",
#         }
#     ],
#     temperature=0.2,
#     top_p=0.7,
#     max_tokens=1024,
#     stream=True,
# )

# for chunk in completion:
#     if chunk.choices[0].delta.content is not None:
#         print(chunk.choices[0].delta.content, end="")


client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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",
)

completion = client.chat.completions.create(
    model="ai21labs/jamba-1.5-large-instruct",
    messages=[
        {
            "role": "user",
            "content": "Write a limerick about the wonders of GPU computing.",
        }
    ],
    temperature=0.2,
    top_p=0.7,
    max_tokens=1024,
    stream=True,
)

for chunk in completion:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
