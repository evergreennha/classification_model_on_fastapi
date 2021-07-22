from libs import *
from predict_model import load_model, img_transforms


app = FastAPI()

@app.get("/")
async def home():
    return "Hello"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    class_names = ["cat","dog","panda"]
    model = load_model()
    img_tranform = img_transforms()
    with torch.no_grad():
        img= Image.open(file.file)
        img_trans = img_tranform(img)
        img_trans = img_trans.unsqueeze(0)
        outputs=model(img_trans)
        result = list(torch.max(outputs,1))
        accuracy = float(result[0][0].numpy())
        class_id = result[1][0].numpy()
        class_name = class_names[class_id]
        dict_result = {"accuracy": accuracy,"class_name":class_name}
    return json.dumps(dict_result, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)