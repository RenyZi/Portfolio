
from django.urls import path
from . import views

urlpatterns = [
    path('',views.home, name='index'),
    path('housepred/', views.houseprediction, name="housepred"),
    path('attritionPred/', views.attritionprediction, name="attritionPred"),
    path('churnPred/', views.churnprediction, name="churnPred"),
    path('fraudPred/', views.fraudprediction, name="fraudPred"),
    path('loandefaultPred/', views.loandefaultprediction, name="loandefaultPred")
 
]
