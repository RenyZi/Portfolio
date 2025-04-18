from django.shortcuts import render
from django.http import request, HttpResponse
from .ml_models.house_predict import HousePricePredictor
from .ml_models.attrition_predict import AttritionPredict
from .ml_models.churn_predict import ChurnPrediction
from .ml_models.fraud_predict import FraudPrediction
from .ml_models.loanDefault_predict import LoanDefaultRisk
import numpy as np




# Create your views here.
def home(request):
    return render(request, 'index.html')

def houseprediction(request):
    house_model = HousePricePredictor()
    house_model.train('lasso')

    # This will always be shown
    dataset = {"data": house_model.generate_data()}

    # Initialize context with dataset
    context = {"data": dataset["data"]}

    if request.method == 'POST':
        Square_feet = int(request.POST.get("square"))
        bedrooms = int(request.POST.get("bedroom"))
        bathrooms = int(request.POST.get("bathroom"))
        houseage = int(request.POST.get("houseage"))
        crimerate = float(request.POST.get("crimerate"))
        distance = float(request.POST.get("city"))

        new_data = np.array([Square_feet, bedrooms, bathrooms, houseage, crimerate, distance])
        predicted_price = house_model.predict(new_data)[0] * 1000

        # Add prediction to the context
        context["predicted"] = f"The predicted house price is $ {predicted_price:.2f}"

    return render(request, 'housepred.html', context)


def attritionprediction(request):
    attrtion_model = AttritionPredict()
    attrtion_model.train('ridge')
    att_dataset = {
        "attrition_dataset": attrtion_model.generator()
    }
    context = {"attrition_dataset": att_dataset['attrition_dataset']}

    # geting features
    if request.method == 'POST':
        Age = int(request.POST.get("Age"))
        job = int(request.POST.get("job"))
        monthlyincome = int(request.POST.get("mincome"))
        load = int(request.POST.get("load"))
        satification = int(request.POST.get("satification"))
        distance = int(request.POST.get("distance"))

        new_data = np.array([Age, job, monthlyincome, load, satification, distance])

        
        context["predicted"] = attrtion_model.predict(new_data)

    return render(request, 'attritionPred.html',context)

def churnprediction(request):
    churn_model = ChurnPrediction()
    churn_model.train()

    dataset = {
        "data": churn_model.dataset()
    }
    churn_context = {"data" : dataset["data"]}

    if request.method == 'POST':
        custage = int(request.POST.get("custage"))
        spending = int(request.POST.get("spending"))
        tenu = int(request.POST.get("tenu"))
        contract = int(request.POST.get("contract"))
        interusage = int(request.POST.get("interusage"))
        support = int(request.POST.get("support"))

        new_features = np.array([custage, spending, tenu, contract, interusage, support])
        prediction = churn_model.predict(new_features)
        churn_context["predicted"] = prediction

    return render(request, 'churnPred.html', churn_context)

def fraudprediction(request):
    fraud = FraudPrediction()
    fraud.train('lasso')
    fraud_table = fraud.dataGenerate()
    fraud_context = {"data": fraud_table}

    if request.method == "POST":
        amount = int(request.POST.get("amount"))
        location = request.POST.get("location")
        day = request.POST.get("day")
        method = request.POST.get("method")
        account = int(request.POST.get("account"))

        # mappings
        location_mapping = {
            "Nairobi": 1, "Nakuru": 2, "Kisumu": 3, "Mombasa": 4, "Kakamega": 5
        }

        time_of_day_mapping = {
            "Morning": 1, "Afternoon": 2, "Evening": 3, "Night": 4
        }

        payment_method_mapping = {
            "Credit Card": 1, "Debit Card": 2, "Paypal": 3, "Bitcoin": 4
        }

        # encode
        location_Encoded = location_mapping[location]
        time_of_day_Encoded = time_of_day_mapping[day]
        payment_method_Encoded = payment_method_mapping[method]

        # prediction input
        new_fraud_features = np.array([
            amount, location_Encoded, time_of_day_Encoded, payment_method_Encoded, account
        ]).astype(int)

        # make prediction
        predicted = fraud.prediction(new_fraud_features)
        fraud_context["predict"] = predicted

    return render(request, 'fraudPred.html', fraud_context)


def loandefaultprediction(request):
    loan_model = LoanDefaultRisk()
    loan_model.train('ridge')
    dataset = {"dataloan": loan_model.generate()}
    cont = {"dataloan": dataset["dataloan"]}

    if request.method == "POST":
        custoage = int(request.POST.get("custoage"))
        anualinc = int(request.POST.get("anualinc"))
        score = int(request.POST.get("score"))
        loanamount = int(request.POST.get("loanamount"))
        loanterm = int(request.POST.get("loanterm"))
        incomeratio = int(loanamount/loanterm)

        new_feature = np.array([custoage, anualinc, score, loanamount, loanterm, incomeratio])
        result = loan_model.prediction(new_feature)
        cont["predicted"] = result 

    return render(request, 'loandefaultPred.html', cont)

