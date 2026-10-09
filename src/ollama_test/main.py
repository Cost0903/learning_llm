from ollama import chat
from ollama import ChatResponse
import os
from dotenv import load_dotenv

load_dotenv()
model = os.getenv("MODEL")
host = os.getenv("HOST")
port = os.getenv("PORT")

# LLM 是文字接龍，會透過設定 temperature 會影響 Sampling
# 他會根據前面的 Context 去往後接龍，接龍的方式會是計算下一個字的可能性，並且根據 Temperature 做權重的計算
# 原始分數 = logit，但 Logit != Probability
# 透過 Softmax 將 Logit 轉換為 Probability
# 最後才進行 Sampling
# P(token_i) = softmax(logit_i / temperature)
# context -> logit -> softmax -> probability -> sampling
# Token 是以單字為單位
# Pulse is a monitoring tool -> 5 個 Token
# 當其中有一個 Token 不同後，就可能造成不同的結果
# 因此 Temperature 在我們做不同用途的事情時，需要的 Temperature 也不同

# 範例:
"""
現在的 Context: A good monitoring tool is 
Temperature = 1 & 2
模型計算出的下一個可能的 Token
Token        Logit(T=0) Temp = 0.5     Temp = 2
-----------------------------------------------------------
Pulse        8.0        8.0 / 0.5 = 16  8.0 / 2 = 4.0            
Monitor      7.5        7.5 / 0.5 = 15  7.5 / 2 = 3.75
Sentry       6.5        6.5 / 0.5 = 13  6.5 / 2 = 3.25
Watch        5.0        5.0 / 0.5 = 10  5.0 / 2 = 2.5
Potato       0.5        0.5 / 0.5 = 1   0.5 / 2 = 0.25

-> 透過 Softmax 將 Logit 轉換為 Probability
Token        Probability(T=0)  T=0.5    T=2
----------------------------------------
Pulse        55%               86%    45%    
Monitor      22%               12%    27%
Sentry       15%               2%     17%
Watch        7%                ~0%    11%
Potato       1%                 0%     0%
總和會是 100%， T=0.5 & T=2 是範例，實際計算方法還須研究
"""
options = {
    "temperature": float(os.getenv("TEMPERATURE")), # 影響候選 Token 的權重，Temperauter 越高，Token 間的權重差距越小
    "top_p": float(os.getenv("TOP_P")) # 影響候選的 Token 範圍，如果是 Top_P=0.5，候選的 Token 根據機率由高往低累積，直到機率總和達到 0.5 為止，再從選出的 Token 中進行 Sampling
}

message = input("Please input your message: ")

# client = ollama.Ollama(host=host, port=port, options=options, model=model)
response: ChatResponse = chat(
    model=model,
    messages=[
        {
            "role": "user",
            "content": message
        },
    ],
    options=options
)

# 大致 LLM Sampling pipeline: 
# Prompt -> Tokenizer -> Transformer 
# -> Logits -> Temperature -> Softmax 
# -> Probability Distribution -> Top-P filtering
# -> Renormalization -> Sampling -> Next Token
# -> 加入 Context -> Repeat Sampling pipeline
print(response.message.content)