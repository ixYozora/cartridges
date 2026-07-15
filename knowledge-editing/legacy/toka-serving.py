import requests

def ask_question(question):
    url = "http://localhost:10210/custom/cartridge/chat/completions"

    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": question}],
        "max_tokens": 1024,
        "cartridges": [
        {
            #twin
            "id": "itachimasoudian-heinrich-heine-university-d-sseldorf/cartridges/s8u3e1vq",
            "source": "wandb"
        },
        {   #korea
            "id": "itachimasoudian-heinrich-heine-university-d-sseldorf/cartridges/co6mesyq",
            "source": "wandb"
        },
        {   # percy snow
            "id": "itachimasoudian-heinrich-heine-university-d-sseldorf/cartridges/4zfmg2vo",
            "source": "wandb"
        },
        {   # icelandic dream
            "id": "itachimasoudian-heinrich-heine-university-d-sseldorf/cartridges/8o1zwwek",
            "source": "wandb"
        }
            # itachimasoudian-heinrich-heine-university-d-sseldorf/cartridges/d2bu2i2b ALL
            # itachimasoudian-heinrich-heine-university-d-sseldorf/cartridges/eul4wdjk ALl2
        ]
    }

    response = requests.post(url, json=payload)

    if response.status_code != 200:
        print(f"Server Error ({response.status_code}): {response.text}")
        return "Error"

    res_json = response.json()
    return res_json['choices'][0]['message']['content']

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['/quit', '/exit']:
            break
        print(f"Assistant: {ask_question(user_input)}")