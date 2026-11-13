from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from helper import predict

app = FastAPI(title="Car Damage Classifier")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")


@app.post("/predict")
async def get_prediction(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    try:
        image_bytes = await file.read()
        
        # Unpack the two returned values from helper.py
        prediction, cost = predict(image_bytes)
        
        # Return both to the frontend
        return {
            "prediction": prediction,
            "estimated_cost": cost
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))