import os
import joblib
from django.apps import AppConfig
from django.conf import settings

class ClassifierConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'classifier'
    
    # Singleton attributes
    model = None
    vectorizer = None

    def ready(self):
        # Ensure that models are only loaded when they exist to prevent crashing on migrations
        model_path = os.path.join(settings.BASE_DIR, 'models', 'model.pkl')
        vectorizer_path = os.path.join(settings.BASE_DIR, 'models', 'vectorizer.pkl')
        
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            # Checking if model is already loaded (can happen in local dev during reload)
            if ClassifierConfig.model is None or ClassifierConfig.vectorizer is None:
                ClassifierConfig.model = joblib.load(model_path)
                ClassifierConfig.vectorizer = joblib.load(vectorizer_path)
                print("ML Model and Vectorizer initialized perfectly in memory.")
