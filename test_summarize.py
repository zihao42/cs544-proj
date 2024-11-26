from summarizer import Summarizer

# 加载 Summarizer 模型
model = Summarizer()

# 输入文本
text = """
Machine learning is a method of data analysis that automates analytical model building. 
It is a branch of artificial intelligence based on the idea that systems can learn from data, 
identify patterns, and make decisions with minimal human intervention. 
The iterative aspect of machine learning is important because as models are exposed to new data, 
they are able to independently adapt.
"""

# 提取关键句，使用 `body` 参数传入文本
key_sentences = model(body=text, num_sentences=2)  # 提取 2 个关键句
print("关键句:")
print(key_sentences)
