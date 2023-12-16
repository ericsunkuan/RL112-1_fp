from openai import OpenAI
client = OpenAI(api_key = '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')

response = client.chat.completions.create(
  model="gpt-4-1106-preview", #-turbo-instruct
  
  
  messages = [
    {
      "role": "system",
      "content": "You are a helpful and expertised assistant."
    },
    {
      "role": "user",
      "content": "目前桃園高鐵站前面有一塊非常大的閒置土地，目前打算開始進行開發，以獲得商業利益同時促進永續發展。請問你對這塊地的利用法法有什麼建議，例如：開發成複合式商場。請同時考慮桃園高鐵站附近有以下優點 ： 1. 高鐵站有許多來自其他縣市的高鐵乘客人流 2. 附近有桃園機場，會有許多國外旅客路經 3. 地點遠離市中心，地價便宜，並盡可能詳細的描述你的每個方案。 "
    }
  ],
      
      

    
  
  max_tokens=2000,
)
print(response)