from rest_framework import serializers

class TextSerializer(serializers.Serializer):
    text = serializers.CharField(required=True, allow_blank=False, min_length=1)
