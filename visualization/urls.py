from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('predict-form', views.predict_form, name='predict_form'),
    path('prediction', views.prediction, name='prediction')
]