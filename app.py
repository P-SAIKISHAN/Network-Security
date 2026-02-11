import sys
import os

import certifi
ca = certifi.where()

from dotenv import load_dotenv
load_dotenv()
mongo_db_url = os.getenv("MONGODB_URL_KEY")
print(mongo_db_url)
import pymongo
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
import pandas as pd

from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel


client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)

from networksecurity.constant.training_pipeline import DATA_INGESTION_COLLECTION_NAME
from networksecurity.constant.training_pipeline import DATA_INGESTION_DATABASE_NAME

database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="./templates")

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training is successful")
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        # Read the uploaded CSV file
        df = pd.read_csv(file.file)
        
        # Define all expected features (30 features total) - ORDER MATTERS!
        expected_features = [
            'having_IP_Address', 'URL_Length', 'Shortining_Service', 'having_At_Symbol',
            'double_slash_redirecting', 'Prefix_Suffix', 'having_Sub_Domain', 'SSLfinal_State',
            'Domain_registeration_length', 'Favicon', 'port', 'HTTPS_token', 'Request_URL',
            'URL_of_Anchor', 'Links_in_tags', 'SFH', 'Submitting_to_email', 'Abnormal_URL',
            'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe', 'age_of_domain',
            'DNSRecord', 'web_traffic', 'Page_Rank', 'Google_Index', 'Links_pointing_to_page',
            'Statistical_report'
        ]
        
        # Add missing columns with default value -1
        missing_features = []
        for feature in expected_features:
            if feature not in df.columns:
                df[feature] = -1
                missing_features.append(feature)
        
        if missing_features:
            logging.info(f"Added missing features with default value -1: {missing_features}")
            print(f"Added missing features: {missing_features}")
        
        # Reorder columns to match the training data order
        df = df[expected_features]
        
        # Load preprocessor and model
        preprocessor = load_object("final_model/preprocessor.pkl")
        final_model = load_object("final_model/model.pkl")
        network_model = NetworkModel(preprocessor=preprocessor, model=final_model)
        
        # Display first row for debugging
        print("First row of data:")
        print(df.iloc[0])
        
        # Make predictions
        y_pred = network_model.predict(df)
        print("Raw Predictions:", y_pred)
        
        # Add predictions to dataframe
        df['predicted_column'] = y_pred
        
        # Map predictions to human-readable labels
        # Handle both -1/1 and 0/1 prediction schemes
        df['prediction_status'] = df['predicted_column'].apply(
            lambda x: 'Legitimate' if x == 1 else 'Phishing'
        )
        
        print("\nPrediction Summary:")
        print(df['prediction_status'].value_counts())
        
        # Create output directory if it doesn't exist
        os.makedirs('prediction_output', exist_ok=True)
        
        # Save predictions to CSV
        df.to_csv('prediction_output/output.csv', index=False)
        print(f"\nPredictions saved to: prediction_output/output.csv")
        
        # Generate HTML table for display
        table_html = df.to_html(classes='table table-striped', index=False)
        
        # Return template with table
        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})
        
    except Exception as e:
        raise NetworkSecurityException(e, sys)

    
if __name__ == "__main__":
    app_run(app, host="0.0.0.0", port=8000)