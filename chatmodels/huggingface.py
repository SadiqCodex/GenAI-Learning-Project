from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1"
)

# Chat wrapper
model = ChatHuggingFace(llm=llm)

# Test
response = model.invoke("who are you?")
print(response.content)

