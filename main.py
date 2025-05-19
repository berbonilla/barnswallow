import os
import io
import base64
import sqlite3
import json
import asyncio
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow import keras
from fastapi import FastAPI, File, UploadFile, Form, WebSocket
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.responses import JSONResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

UPLOAD_FOLDER = "uploads"
MODEL_FOLDER = "models"
INFO_FOLDER = "info"  # Added the info folder
DB_FILE = "uploads.db"
HTML_FILE = "gen_report.html"

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ Corrected the directory paths (absolute)
MODEL_DIR = os.path.join(BASE_DIR, "models")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
STATIC_DIR = os.path.join(BASE_DIR, "static")
HTML_PATH = os.path.join(BASE_DIR, "gen_report.html")
IMAGES_DIR = os.path.join(BASE_DIR, "Images")

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)


SEQ_LEN = 5

DIRECTION_MAP = {0: 'N', 1: 'NE', 2: 'E', 3: 'SE', 4: 'S', 5: 'SW', 6: 'NW'}
BIRD_DENSITY_MAP = {0: 'No Birds', 1: 'Small Amount Expected', 2: 'Large Group Expected'}

features = [
    'Latitude', 'Longitude', 'Temperature', 'Wind Speed',
    'Direction Code', 'Bird Density',
    'Arrival_min', 'Dep_first_min', 'Dep_last_min',
    'MapLoc_enc', 'DayOfYear'
]
target = [
    'Latitude', 'Longitude', 'Temperature', 'Wind Speed',
    'Direction Code', 'Bird Density',
    'Arrival_min', 'Dep_first_min', 'Dep_last_min'
]

# ✅ Mount static folder using absolute path
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/Images", StaticFiles(directory=IMAGES_DIR), name="images")

@app.get("/", response_class=HTMLResponse)
def root():
    if os.path.exists(HTML_PATH):
        with open(HTML_PATH, encoding="utf-8") as f:
            return HTMLResponse(f.read())
    else:
        return HTMLResponse(
            content="<html><body><h1>Fallback</h1></body></html>",
            status_code=200
        )
def minutes_to_hhmm(minutes):
    minutes = int(round(minutes))
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours:02d}:{mins:02d}"

def direction_label_from_code(code):
    code_int = int(round(code))
    return DIRECTION_MAP.get(code_int, f"Unknown({code_int})")

def bird_density_label(code):
    code_int = int(round(code))
    return BIRD_DENSITY_MAP.get(code_int, f"Unknown({code_int})")

def time_to_minutes(t):
    if pd.isnull(t): return 0
    t = str(t)
    try:
        h, m, *_ = t.split(':')
        return int(h) * 60 + int(m)
    except:
        return 0

@app.post("/forecast")
def forecast(
    model_name: str = Form(...),
    data_name: str = Form(...),
    num_days: int = Form(...)
):
    model_path = os.path.join(MODEL_DIR, model_name)
    data_path = os.path.join(UPLOAD_DIR, data_name)

    if not os.path.exists(model_path):
        return JSONResponse({"error": f"Model file not found: {model_path}"}, status_code=404)
    if not os.path.isfile(model_path):
        return JSONResponse({"error": f"Model path is not a file: {model_path}"}, status_code=400)
        
    if not os.path.exists(data_path):
        return JSONResponse({"error": f"Data file not found: {data_path}"}, status_code=404)
    if not os.path.isfile(data_path):
        return JSONResponse({"error": f"Data path is not a file: {data_path}"}, status_code=400)

    # --- Data prep ---
    try:
        df = pd.read_excel(data_path)
    except Exception as ex:
        return JSONResponse({"error": f"Invalid or corrupted xlsx file '{data_name}': {str(ex)}"}, status_code=400)

    # Check for required columns for preprocessing and feature generation
    required_base_cols = ['Time of Arrival', 'Time of Departure of First Flock', 
                          'Time of Departure of Last Flock', 'Map Location', 'Date']
    # Core features that are expected to be directly in the excel and used for scaling
    required_feature_cols = ['Latitude', 'Longitude', 'Temperature', 'Wind Speed', 'Direction Code', 'Bird Density']

    missing_base_cols = [col for col in required_base_cols if col not in df.columns]
    if missing_base_cols:
        return JSONResponse({"error": f"Data file '{data_name}' is missing base column(s): {', '.join(missing_base_cols)} needed for preprocessing."}, status_code=400)

    missing_feature_cols = [col for col in required_feature_cols if col not in df.columns]
    if missing_feature_cols:
         return JSONResponse({"error": f"Data file '{data_name}' is missing core feature column(s): {', '.join(missing_feature_cols)} required for scaling."}, status_code=400)

    try:
        df['Arrival_min'] = df['Time of Arrival'].apply(time_to_minutes)
        df['Dep_first_min'] = df['Time of Departure of First Flock'].apply(time_to_minutes)
        df['Dep_last_min'] = df['Time of Departure of Last Flock'].apply(time_to_minutes)
        
        # Handle potential empty or all-NaN 'Map Location' which can cause issues with LabelEncoder
        if df['Map Location'].isnull().all():
             return JSONResponse({"error": f"Column 'Map Location' in '{data_name}' is empty or contains all NaN values."}, status_code=400)
        if df['Map Location'].nunique() == 0 : # If there are no unique non-null values
            return JSONResponse({"error": f"Column 'Map Location' in '{data_name}' has no valid unique values to encode."}, status_code=400)

        le = LabelEncoder()
        df['MapLoc_enc'] = le.fit_transform(df['Map Location'].astype(str)) # astype(str) to handle mixed types or NaNs robustly before encoding
        df['DayOfYear'] = pd.to_datetime(df['Date']).dt.dayofyear
    except KeyError as e: # Should be caught by earlier checks, but as a safeguard
        return JSONResponse({"error": f"Data file '{data_name}' is missing column: {str(e)} needed for preprocessing."}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"Error during data preprocessing for '{data_name}': {str(e)}"}, status_code=400)

    try:
        scaler = MinMaxScaler()
        df_scaled = df.copy()
        # Ensure all features defined in the global features list are present in df before scaling
        missing_scaling_features = [f_col for f_col in features if f_col not in df.columns]
        if missing_scaling_features:
            return JSONResponse({"error": f"DataFrame is missing features required for scaling: {', '.join(missing_scaling_features)}. Expected all of: {features}"}, status_code=400)
        
        df_scaled[features] = scaler.fit_transform(df[features])
    except KeyError as e: # Should be caught by earlier checks
        return JSONResponse({"error": f"Data file '{data_name}' is missing column: {str(e)} needed for scaling. Expected features: {features}"}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"Error during data scaling for '{data_name}': {str(e)}"}, status_code=400)

    if len(df_scaled) < SEQ_LEN:
        return JSONResponse({"error": f"Data file '{data_name}' has insufficient data rows ({len(df_scaled)}) to form a sequence of length {SEQ_LEN}."}, status_code=400)
    
    start_idx = len(df_scaled) - SEQ_LEN
    seq = df_scaled.iloc[start_idx:start_idx+SEQ_LEN][features].values

    # --- Load Model ---
    try:
        model = keras.models.load_model(model_path)
    except Exception as ex:
        return JSONResponse({"error": f"Error loading model '{model_name}': {str(ex)}"}, status_code=400)

    # --- Forecast Loop ---
    forecasted = []
    current_seq = seq.copy()
    for day_num in range(num_days):
        try:
            pred = model.predict(current_seq[np.newaxis, :, :])[0]
        except Exception as e:
            return JSONResponse({"error": f"Error during model prediction on day {day_num+1}: {str(e)}"}, status_code=500)
        
        forecasted.append(pred.copy())
        next_input = current_seq[-1].copy() # Start with a copy of the last known (or predicted) input
        
        # Update the features that are part of the target prediction
        for i, col_name in enumerate(target):
            feat_idx = features.index(col_name) # Find index in the full 'features' list
            next_input[feat_idx] = pred[i]      # Update with the new prediction

        # For features not in 'target', we need a strategy.
        # Current approach: they are implicitly carried over from the last step of current_seq.
        # If 'DayOfYear' or 'MapLoc_enc' were to change predictably, that logic would go here.
        # Example: if DayOfYear should increment:
        # if 'DayOfYear' in features and 'DayOfYear' not in target:
        #     doy_idx = features.index('DayOfYear')
        #     last_doy_scaled = next_input[doy_idx]
        #     # This is tricky because it's scaled. We'd need to inverse, increment, then re-scale.
        #     # For simplicity, this example assumes it's carried over or handled by the model implicitly if it's a target.
        #     # If DayOfYear is crucial and not a target, the model architecture should be designed to handle time.

        current_seq = np.vstack([current_seq[1:], next_input])
    forecasted = np.array(forecasted)

    # --- Inverse transform ---
    def inv_transform_predictions(preds_scaled, scaler_obj, all_features_list, target_features_list):
        inversed_preds = []
        for row_scaled in preds_scaled:
            # Create a dummy array matching the shape/order of all_features_list for inverse_transform
            dummy_full_features_row = np.zeros((1, len(all_features_list)))
            
            # Populate the dummy array with predicted values at correct positions
            for i, target_col_name in enumerate(target_features_list):
                target_idx_in_all = all_features_list.index(target_col_name)
                dummy_full_features_row[0, target_idx_in_all] = row_scaled[i]
            
            # Inverse transform the dummy row
            inversed_full_features_row = scaler_obj.inverse_transform(dummy_full_features_row)[0]
            
            # Extract only the target features from the inversed full row
            inversed_target_values = [inversed_full_features_row[all_features_list.index(t)] for t in target_features_list]
            inversed_preds.append(inversed_target_values)
        return np.array(inversed_preds)

    try:
        forecasted_inv = inv_transform_predictions(forecasted, scaler, features, target)
    except Exception as e:
        return JSONResponse({"error": f"Error during inverse transformation of results: {str(e)}"}, status_code=500)

    # --- Format result ---
    result_rows = []
    for i, row_data in enumerate(forecasted_inv):
        if len(row_data) != len(target):
            return JSONResponse({"error": f"Mismatch between target feature count ({len(target)}) and forecasted data elements ({len(row_data)}) on day {i+1} after inverse transform."}, status_code=500)

        # Unpack based on the order in the 'target' list
        unpacked_data = dict(zip(target, row_data))

        direction_code_val = unpacked_data['Direction Code']
        bird_density_val = unpacked_data['Bird Density']
        
        direction_code_int = int(round(direction_code_val))
        bird_density_int = int(round(bird_density_val))
        if bird_density_int not in BIRD_DENSITY_MAP.keys():
            bird_density_int = 0 

        res = {
            "Preditiction": i + 1,
            "Latitude": unpacked_data['Latitude'],
            "Longitude": unpacked_data['Longitude'],
            "Temperature": round(unpacked_data['Temperature'], 2),
            "Wind Speed": round(unpacked_data['Wind Speed'], 2),
            "Direction Code": direction_code_int,
            "Direction": direction_label_from_code(direction_code_val),
            "Bird Density Code": bird_density_int,
            "Bird Density": bird_density_label(bird_density_int),
            "Time of Arrival": minutes_to_hhmm(unpacked_data['Arrival_min']),
            "Departure First Flock": minutes_to_hhmm(unpacked_data['Dep_first_min']),
            "Departure Last Flock": minutes_to_hhmm(unpacked_data['Dep_last_min'])
        }
        result_rows.append(res)
    return JSONResponse({"results": result_rows})


@app.get("/list_assets")
def list_assets():
    models = []
    datafiles = []
    
    try:
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
            print(f"INFO: Created directory: {MODEL_DIR}")
        elif not os.path.isdir(MODEL_DIR):
            return JSONResponse({"error": f"Path '{MODEL_DIR}' exists but is not a directory."}, status_code=500)
        
        model_files = os.listdir(MODEL_DIR)
        models = [f for f in model_files if f.endswith((".keras", ".h5"))] 
        
        if not os.path.exists(UPLOAD_DIR):
            os.makedirs(UPLOAD_DIR)
            print(f"INFO: Created directory: {UPLOAD_DIR}")
        elif not os.path.isdir(UPLOAD_DIR):
            return JSONResponse({"error": f"Path '{UPLOAD_DIR}' exists but is not a directory."}, status_code=500)

        data_files_list = os.listdir(UPLOAD_DIR)
        datafiles = [f for f in data_files_list if f.endswith(".xlsx")]
            
        return {"models": models, "datafiles": datafiles}
    except Exception as ex:
        return JSONResponse({"error": f"Error listing assets: {str(ex)}"}, status_code=500)



# Global variable to store training status for WebSocket updates
training_status = {"active": False, "step": "", "progress": 0, "total": 0}

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS uploads (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, 
                        file_name TEXT, 
                        saved_name TEXT, 
                        uploaded_at TEXT,
                        file_size INTEGER,
                        columns INTEGER,
                        rows INTEGER,
                        version INTEGER DEFAULT 1);""")
    
    cursor.execute("""CREATE TABLE IF NOT EXISTS models (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT,
                        file_path TEXT,
                        created_at TEXT,
                        file_size INTEGER,
                        version INTEGER DEFAULT 1,
                        metrics TEXT,
                        upload_id INTEGER,
                        FOREIGN KEY (upload_id) REFERENCES uploads(id));""")
    conn.commit()
    conn.close()

init_db()

def update_training_status(step, progress=None, total=None):
    global training_status
    training_status["step"] = step
    if progress is not None:
        training_status["progress"] = progress
    if total is not None:
        training_status["total"] = total
    
    # Print to server console for debugging
    if progress is not None and total is not None:
        percentage = (progress / total) * 100 if total > 0 else 0
        print(f"Training: {step} - {progress}/{total} ({percentage:.2f}%)")
    else:
        print(f"Training: {step}")


def save_upload(file: UploadFile):
    timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S_%f')
    saved_name = f"{file.filename}"
    saved_path = os.path.join(UPLOAD_FOLDER, saved_name)
    
    # Save file in the uploads folder
    with open(saved_path, "wb") as f:
        file_content = file.file.read()
        f.write(file_content)
        file_size = len(file_content)
    
    update_training_status("Analyzing uploaded file...")

    # Get row and column count
    try:
        df = pd.read_excel(saved_path)
        rows = len(df)
        columns = len(df.columns)
    except:
        rows = 0
        columns = 0
    
    # Save metadata info to the "info" folder
    info_file_name = f"{timestamp}_info.json"
    info_file_path = os.path.join(INFO_FOLDER, info_file_name)
    file_info = {
        "file_name": file.filename,
        "saved_name": saved_name,
        "uploaded_at": datetime.utcnow().isoformat(),
        "file_size": file_size,
        "columns": columns,
        "rows": rows
    }
    
    os.makedirs(INFO_FOLDER, exist_ok=True)  # Make sure info folder exists
    
    with open(info_file_path, "w") as f:
        json.dump(file_info, f, indent=4)

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO uploads (file_name, saved_name, uploaded_at, file_size, columns, rows) VALUES (?, ?, ?, ?, ?, ?)",
        (file.filename, saved_name, datetime.utcnow().isoformat(), file_size, columns, rows)
    )
    upload_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return saved_path, saved_name, upload_id

def save_model(model_path, metrics, upload_id):
    file_size = os.path.getsize(model_path)
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Generate a proper name for the model
    cursor.execute("SELECT file_name FROM uploads WHERE id = ?", (upload_id,))
    upload_name = cursor.fetchone()[0].replace('.xlsx', '')
    
    # Check if there's already a model for this upload
    cursor.execute("SELECT COUNT(*) FROM models WHERE upload_id = ?", (upload_id,))
    count = cursor.fetchone()[0]
    i = 1
    version = count + i
    
    model_name = f"{upload_name}_model_v{version}"
    
    # Save model in the models folder
    model_filename = f"{model_name}.keras"
    model_path = os.path.join(MODEL_FOLDER, model_filename)

    cursor.execute(
        "INSERT INTO models (name, file_path, created_at, file_size, version, metrics, upload_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (model_name, model_path, datetime.utcnow().isoformat(), file_size, version, json.dumps(metrics), upload_id)
    )
    model_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return model_id

def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64

class VerboseCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        update_training_status(f"Training epoch {epoch+1}/{self.params['epochs']}", epoch+1, self.params['epochs'])
        
    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        update_training_status(f"Epoch {epoch+1}/{self.params['epochs']} completed - loss: {loss:.4f}, val_loss: {val_loss:.4f}", epoch+1, self.params['epochs'])

def run_pipeline(FILE, upload_id):
    global training_status
    training_status = {"active": True, "step": "Starting pipeline...", "progress": 0, "total": 10}
    
    update_training_status("Loading data...", 1, 10)
    df = pd.read_excel(FILE)

    update_training_status("Preprocessing time features...", 2, 10)
    def time_to_minutes(t):
        if pd.isnull(t): return 0
        t = str(t)
        try:
            h, m, *_ = t.split(':')
            return int(h) * 60 + int(m)
        except:
            return 0

    df['Arrival_min'] = df['Time of Arrival'].apply(time_to_minutes)
    df['Dep_first_min'] = df['Time of Departure of First Flock'].apply(time_to_minutes)
    df['Dep_last_min'] = df['Time of Departure of Last Flock'].apply(time_to_minutes)
    
    update_training_status("Encoding categorical features...", 3, 10)
    le = LabelEncoder()
    df['MapLoc_enc'] = le.fit_transform(df['Map Location'])
    df['DayOfYear'] = pd.to_datetime(df['Date']).dt.dayofyear

    features = [
        'Latitude', 'Longitude',
        'Temperature', 'Wind Speed',
        'Direction Code', 'Bird Density',
        'Arrival_min', 'Dep_first_min', 'Dep_last_min',
        'MapLoc_enc', 'DayOfYear'
    ]
    target = [
        'Latitude', 'Longitude', 'Temperature', 'Wind Speed',
        'Direction Code', 'Bird Density',
        'Arrival_min', 'Dep_first_min', 'Dep_last_min'
    ]

    update_training_status("Calculating correlation matrix...", 4, 10)
    corr = df[features].corr()
    fig = plt.figure(figsize=(12,8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title("Feature Correlation Matrix")
    corr_plot = plot_to_base64(fig)
    corr.to_csv("correlation_matrix.csv")

    update_training_status("Normalizing features...", 5, 10)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])
    df_scaled = pd.DataFrame(scaled, columns=features)

    update_training_status("Creating sequences for LSTM...", 6, 10)
    SEQ_LEN = 5
    def make_sequences(df, seq_len=5):
        X, Y = [], []
        for i in range(len(df)-seq_len):
            X.append(df.iloc[i:i+seq_len][features].values)
            Y.append(df.iloc[i+seq_len][target].values)
        return np.array(X), np.array(Y)
    X, y = make_sequences(df_scaled, SEQ_LEN)

    if len(X) < 10:
        training_status["active"] = False
        return {"error": "Not enough rows after sequence conversion!"}

    update_training_status("Splitting data into train/validation/test sets...", 7, 10)
    n_total = X.shape[0]
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.15)
    n_test = n_total - n_train - n_val

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]

    update_training_status("Building LSTM model...", 8, 10)
    tf.keras.backend.clear_session()
    model = keras.Sequential([
        keras.layers.Input(shape=(SEQ_LEN, len(features))),
        keras.layers.LSTM(64),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(len(target))
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    update_training_status("Training model...", 8, 10)
    verbose_callback = VerboseCallback()
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop, verbose_callback],
        verbose=0
    )
    
    # Generate model filename based on upload_id
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM models WHERE upload_id = ?", (upload_id,))
    version = cursor.fetchone()[0] + 1
    conn.close()
    
    update_training_status("Saving model...", 9, 10)
    model_filename = f"model_{upload_id}_v{version}.keras"
    model_path = os.path.join(MODEL_FOLDER, model_filename)
    model.save(model_path)

    update_training_status("Evaluating model...", 10, 10)
    y_pred = model.predict(X_test, verbose=0)

    def inv_transform(preds):
        inv = np.zeros((len(preds), len(features)))
        idxs = [features.index(f) for f in target]
        inv[:, idxs] = preds
        rec = scaler.inverse_transform(inv)
        return rec[:, idxs]
    y_test_inv = inv_transform(y_test)
    y_pred_inv = inv_transform(y_pred)

    metrics = []
    metrics_dict = {}
    for i, col in enumerate(target):
        mae = mean_absolute_error(y_test_inv[:,i], y_pred_inv[:,i])
        rmse = np.sqrt(mean_squared_error(y_test_inv[:,i], y_pred_inv[:,i]))
        mse = mean_squared_error(y_test_inv[:,i], y_pred_inv[:,i])
        metrics.append({"feature": col, "MAE": mae, "RMSE": rmse, "MSE": mse})
        metrics_dict[col] = {"MAE": mae, "RMSE": rmse, "MSE": mse}

    update_training_status("Generating plots...", 10, 10)
    fig = plt.figure(figsize=(18, 12))
    for i, col in enumerate(target):
        plt.subplot(3, 3, i+1)
        plt.plot(y_test_inv[:50, i], label='Actual')
        plt.plot(y_pred_inv[:50, i], label='Predicted')
        plt.title(col)
        if i==0: plt.legend()
    plt.tight_layout()
    avp_plot = plot_to_base64(fig)

    fig = plt.figure()
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend(); plt.title('Loss history'); plt.xlabel('Epoch')
    loss_plot = plot_to_base64(fig)
    
    # Save model to database with metrics
    model_id = save_model(model_path, metrics_dict, upload_id)

    update_training_status("Pipeline completed!", 10, 10)
    training_status["active"] = False

    return {
        "model_id": model_id,
        "metrics": metrics,
        "corr_plot": corr_plot,
        "avp_plot": avp_plot,
        "loss_plot": loss_plot,
    }

@app.websocket("/ws/training-status")
async def websocket_training_status(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Send current training status
            await websocket.send_json(training_status)
            await asyncio.sleep(0.5)  # Update every half second
    except Exception as e:
        print(f"WebSocket error: {e}")


@app.get("/uploads")
async def get_uploads():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT id, file_name, saved_name, uploaded_at, file_size, columns, rows, version FROM uploads ORDER BY id DESC")
    uploads = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return uploads if uploads else []

@app.get("/models")
async def get_models():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""SELECT id, name, file_path, created_at, file_size, version, metrics FROM models ORDER BY id DESC""")
    models = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return models if models else []


@app.get("/model/{model_id}")
def get_model(model_id: int):
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT m.id, m.name, m.file_path, m.created_at, m.file_size, m.version, m.metrics,
               u.file_name as upload_name, m.upload_id
        FROM models m
        JOIN uploads u ON m.upload_id = u.id
        WHERE m.id = ?
    """, (model_id,))
    model = dict(cursor.fetchone())
    model['metrics'] = json.loads(model['metrics'])
    conn.close()
    return model

@app.post("/rename/upload/{upload_id}")
async def rename_upload(upload_id: int, name: str = Form(...)):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("UPDATE uploads SET file_name = ? WHERE id = ?", (name, upload_id))
    conn.commit()
    conn.close()
    return {"success": True}

@app.post("/rename/model/{model_id}")
async def rename_model(model_id: int, name: str = Form(...)):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("UPDATE models SET name = ? WHERE id = ?", (name, model_id))
    conn.commit()
    conn.close()
    return {"success": True}

@app.delete("/delete/upload/{upload_id}")
async def delete_upload(upload_id: int):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Get filename to delete from disk
    cursor.execute("SELECT saved_name FROM uploads WHERE id = ?", (upload_id,))
    saved_name = cursor.fetchone()[0]
    
    # Delete from database
    cursor.execute("DELETE FROM uploads WHERE id = ?", (upload_id,))
    cursor.execute("DELETE FROM models WHERE upload_id = ?", (upload_id,))
    conn.commit()
    conn.close()
    
    # Delete the file from disk if it exists
    file_path = os.path.join(UPLOAD_FOLDER, saved_name)
    if os.path.exists(file_path):
        os.remove(file_path)
        
    return {"success": True}

@app.delete("/delete/model/{model_id}")
async def delete_model(model_id: int):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Get filename to delete from disk
    cursor.execute("SELECT file_path FROM models WHERE id = ?", (model_id,))
    file_path = cursor.fetchone()[0]
    
    # Delete from database
    cursor.execute("DELETE FROM models WHERE id = ?", (model_id,))
    conn.commit()
    conn.close()
    
    # Delete the file from disk if it exists
    if os.path.exists(file_path):
        os.remove(file_path)
        
    return {"success": True}

@app.get("/download/upload/{upload_id}")
async def download_upload(upload_id: int):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT file_name, saved_name FROM uploads WHERE id = ?", (upload_id,))
    result = cursor.fetchone()
    conn.close()
    
    if not result:
        return {"error": "File not found"}
    
    file_name, saved_name = result
    file_path = os.path.join(UPLOAD_FOLDER, saved_name)
    
    if not os.path.exists(file_path):
        return {"error": "File not found on disk"}
        
    return FileResponse(path=file_path, filename=file_name, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

@app.get("/download/model/{model_id}")
async def download_model(model_id: int):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT name, file_path FROM models WHERE id = ?", (model_id,))
    result = cursor.fetchone()
    conn.close()
    
    if not result:
        return {"error": "Model not found"}
    
    model_name, file_path = result
    
    if not os.path.exists(file_path):
        return {"error": "Model file not found on disk"}
        
    return FileResponse(path=file_path, filename=f"{model_name}.keras", media_type='application/octet-stream')

LAST_RESULTS = None

@app.get("/", response_class=HTMLResponse)
def index():
    with open(HTML_FILE, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/upload", response_class=HTMLResponse)
async def upload(file: UploadFile = File(...)):
    global LAST_RESULTS
    saved_path, _, upload_id = save_upload(file)
    LAST_RESULTS = run_pipeline(saved_path, upload_id)
    return RedirectResponse(url="/?tab=validation&result=1", status_code=303)

@app.get("/last_results")
async def get_last_results():
    global LAST_RESULTS
    if LAST_RESULTS is None:
        return {"error": "No results yet. Upload a file first."}
    return LAST_RESULTS


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8001))  # default to 8001 locally if you want
    uvicorn.run("main:app", host="0.0.0.0", port=port)
