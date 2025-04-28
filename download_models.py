import gdown
import os

# Ensure models folder exists
os.makedirs('/app/models', exist_ok=True)

# Danh sách các models
models = [
    {
        "file_id": "1APfWKSS-lHpI5yERsxD2_FZSntGwUqWT",
        "confirm_token": "pbef",
        "output": "/app/models/card_detect.pt"
    },
    {
        "file_id": "1cMNwpR9m4QAwv2lK2QXZvGaqgTg904lh",
        "confirm_token": "pbef",
        "output": "/app/models/face_card_detect.pt"
    },
    {
        "file_id": "1UUXsI_Y1BAiPQ3wuk2gyFfMWhzeLbyJy",
        "confirm_token": "pbef",
        "output": "/app/models/head_detect.pt"
    },
    {
        "file_id": "1xWOiSHxe_QBzdzYmQ0IM9wWTQU7A5c4n",
        "confirm_token": "pbef",
        "output": "/app/models/text_recog.pt"
    }
]

# Tải từng model
for model in models:
    url = f"https://drive.google.com/uc?export=download&confirm={model['confirm_token']}&id={model['file_id']}"
    print(f"Downloading {model['output']}...")
    gdown.download(url, model['output'], quiet=False)
