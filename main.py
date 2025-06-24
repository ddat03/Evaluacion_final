from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from data_service import (
    initialize_service, 
    get_model_names, 
    predict_churn,
    get_metrics,
    get_confusion_matrix,
    get_feature_importance,
    get_eda_data,
    get_customer_insights,
    get_health_status
)

# Crear app FastAPI
app = FastAPI(title="Telco Customer Churn Predictor", version="1.0.0")
templates = Jinja2Templates(directory="templates")

# ============================================================================
# RUTA PRINCIPAL
# ============================================================================

@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    """Página principal con el formulario"""
    return templates.TemplateResponse("form.html", {
        "request": request,
        "model_names": get_model_names()
    })

# ============================================================================
# PREDICCIÓN
# ============================================================================

@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    model_name: str = Form(...),
    version: str = Form("completa"),
    # Características numéricas
    SeniorCitizen: int = Form(0),
    tenure: int = Form(1),
    MonthlyCharges: float = Form(50.0),
    TotalCharges: float = Form(100.0),
    # Características categóricas
    gender: str = Form("Male"),
    Partner: str = Form("No"),
    Dependents: str = Form("No"),
    PhoneService: str = Form("Yes"),
    MultipleLines: str = Form("No"),
    InternetService: str = Form("DSL"),
    OnlineSecurity: str = Form("No"),
    OnlineBackup: str = Form("No"),
    DeviceProtection: str = Form("No"),
    TechSupport: str = Form("No"),
    StreamingTV: str = Form("No"),
    StreamingMovies: str = Form("No"),
    Contract: str = Form("Month-to-month"),
    PaperlessBilling: str = Form("Yes"),
    PaymentMethod: str = Form("Electronic check")
):
    """Realizar predicción de churn"""
    # Preparar datos del cliente
    customer_data = {
        'SeniorCitizen': SeniorCitizen,
        'tenure': tenure,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges,
        'gender': gender,
        'Partner': Partner,
        'Dependents': Dependents,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod
    }
    
    # Realizar predicción
    result = predict_churn(customer_data, model_name, version)
    
    # Preparar mensaje de resultado
    if result["success"]:
        prediction_text = result["prediction_label"]
        probability = result["probabilities"]["churn"] * 100
        features_used = "reducidas (7)" if version == "reducida" else "completas (19)"
        
        result_message = f"""
        🎯 Modelo: {model_name}
        📊 Features: {features_used}
        🔮 Predicción: {prediction_text}
        📈 Probabilidad de Churn: {probability:.1f}%
        """
    else:
        result_message = f"❌ Error: {result['error']}"
    
    return templates.TemplateResponse("form.html", {
        "request": request,
        "model_names": get_model_names(),
        "result": result_message
    })

# ============================================================================
# APIs PARA DASHBOARD (OPCIONAL)
# ============================================================================

@app.get("/api/predict")
def api_predict(
    model_name: str,
    version: str = "completa",
    SeniorCitizen: int = 0,
    tenure: int = 1,
    MonthlyCharges: float = 50.0,
    TotalCharges: float = 100.0,
    gender: str = "Male",
    Partner: str = "No",
    Dependents: str = "No",
    PhoneService: str = "Yes",
    MultipleLines: str = "No",
    InternetService: str = "DSL",
    OnlineSecurity: str = "No",
    OnlineBackup: str = "No",
    DeviceProtection: str = "No",
    TechSupport: str = "No",
    StreamingTV: str = "No",
    StreamingMovies: str = "No",
    Contract: str = "Month-to-month",
    PaperlessBilling: str = "Yes",
    PaymentMethod: str = "Electronic check"
):
    """API endpoint para predicción (GET)"""
    customer_data = {
        'SeniorCitizen': SeniorCitizen, 'tenure': tenure, 'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges, 'gender': gender, 'Partner': Partner,
        'Dependents': Dependents, 'PhoneService': PhoneService, 'MultipleLines': MultipleLines,
        'InternetService': InternetService, 'OnlineSecurity': OnlineSecurity, 'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection, 'TechSupport': TechSupport, 'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies, 'Contract': Contract, 'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod
    }
    
    result = predict_churn(customer_data, model_name, version)
    return JSONResponse(result)

@app.get("/api/metrics/{model_name}")
def get_model_metrics(model_name: str):
    """Obtener métricas del modelo"""
    return JSONResponse(get_metrics(model_name))

@app.get("/api/confusion_matrix/{model_name}")
def get_model_confusion_matrix(model_name: str):
    """Obtener matriz de confusión"""
    return JSONResponse(get_confusion_matrix(model_name))

@app.get("/api/feature_importance/{model_name}")
def get_model_feature_importance(model_name: str):
    """Obtener importancia de características"""
    return JSONResponse(get_feature_importance(model_name))

@app.get("/api/insights")
def get_business_insights():
    """Obtener insights de negocio"""
    return JSONResponse(get_customer_insights())

@app.get("/api/models")
def get_available_models():
    """Obtener modelos disponibles"""
    return JSONResponse({"models": get_model_names()})

@app.get("/health")
def health_check():
    """Health check"""
    return JSONResponse(get_health_status())

# ============================================================================
# INICIALIZACIÓN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Inicializar aplicación"""
    print("🚀 Iniciando Telco Customer Churn Predictor...")
    if initialize_service():
        print("✅ Aplicación lista!")
    else:
        print("❌ Error en inicialización")

# ============================================================================
# EJECUTAR
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)