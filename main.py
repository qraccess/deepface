from deepface import DeepFace
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()


class ImageData(BaseModel):
    image: str  # base64 encoded string


@app.post("/facedetect")
async def face_detect(data: ImageData):
    # 解码 base64 图像
    ret = {
        "code": 200,
        "msg": "",
        "data": {
            "facecount": 0,
            "gender": [],
            "age": []
        }
    }
    input_image = data.image
    if not data.image.startswith('data:image/') and not data.image.startswith('http'):
        input_image = 'data:image/jpeg;base64,' + data.image
    try:
        objs = DeepFace.analyze(img_path=input_image,
                                actions=['age', 'gender']
                                )
        ret['data']['facecount'] = len(objs)
        ret['data']['gender'] = [obj['dominant_gender'].lower()
                                 for obj in objs]
        ret['data']['age'] = [obj['age'] for obj in objs]
        ret['msg'] = f'{len(objs)} face'
    except Exception:
        ret['msg'] = 'no face'
    # 在这里，你可以使用你的面部检测代码处理图像

    return ret


@app.get("/")
def read_root():
    return {"Hello": "World"}
