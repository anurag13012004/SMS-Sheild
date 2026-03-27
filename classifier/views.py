from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.apps import apps
from .serializers import TextSerializer
import re

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

class ClassifyView(APIView):
    def post(self, request):
        serializer = TextSerializer(data=request.data)
        if serializer.is_valid():
            text = serializer.validated_data['text']
            
            # Access the loaded singleton models
            classifier_app = apps.get_app_config('classifier')
            model = classifier_app.model
            vectorizer = classifier_app.vectorizer
            
            if model is None or vectorizer is None:
                return Response(
                    {"error": "ML Model and Vectorizer are not loaded."},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            
            # Preprocess, vectorize, and predict
            clean_text = preprocess_text(text)
            text_vec = vectorizer.transform([clean_text])
            
            prediction = model.predict(text_vec)[0]  # will be 'spam' or 'ham'
            label = prediction.capitalize()
            
            # Get the confidence score of the prediction
            probabilities = model.predict_proba(text_vec)[0]
            confidence = max(probabilities)
            
            return Response({
                "text": text,
                "label": label,
                "confidence": round(confidence, 4),
                "status": "success"
            }, status=status.HTTP_200_OK)
            
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
