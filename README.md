# House Price Regression Service

A trainable house-price regression project with a CSV training workflow, a saved model artifact, and a FastAPI prediction endpoint. It replaces the former fixed price formula with a model fit on observed housing data.

## Quick start

Python 3.11 or later is recommended.

~~~bash
python -m venv .venv
# Activate the environment, then:
pip install -r requirements.txt
python price_predictor.py train --dataset california
uvicorn app:app --reload
~~~

The first training command downloads and caches scikit-learn's California housing dataset. It contains 20,640 records with eight numeric features. The target is median house value in units of 100,000 USD. Training reports holdout MAE, RMSE, and R-squared, then saves the final model at models/house_price_model.joblib.

## Call the prediction API

After training and starting the service, send feature values with the names used by the model.

~~~bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data '{"features":{"MedInc":8.3,"HouseAge":21,"AveRooms":6.9,"AveBedrms":1.0,"Population":300,"AveOccup":3.0,"Latitude":37.8,"Longitude":-122.4}}'
~~~

The response includes predicted_price, target_name, and target_units. The service also exposes /health and interactive API documentation at /docs.

## Train from your own CSV

Provide a CSV with a numeric target and one or more feature columns. Numeric and categorical features are supported; missing feature values are imputed, and unseen categorical values are ignored at prediction time.

~~~bash
python price_predictor.py train --data data/houses.csv --target SalePrice --target-units USD
python price_predictor.py predict --model models/house_price_model.joblib --data data/new_homes.csv --output predictions.csv
~~~

The training command reserves a holdout split for evaluation, prints MAE, RMSE, and R-squared, then fits the saved artifact on all supplied rows. Prediction inputs must include the feature columns used during training. Extra columns are ignored, and predictions are written to the predicted_price column.

## Docker

Train a model first so models/house_price_model.joblib exists. Then build and run the API:

~~~bash
docker build -t house-price-service .
docker run --rm -p 8000:8000 house-price-service
~~~

Set MODEL_PATH to use a different artifact path.

## Model and data notes

The California dataset is a public benchmark, not a promise that its estimates transfer to other housing markets. For local use with a different region, train on representative, properly licensed data and evaluate on a held-out sample from that market. The project reports model metrics; it does not claim production readiness, valuation accuracy, or fairness without market-specific evidence.

Model artifacts use joblib serialization. Only load artifacts you trained or otherwise trust. The example API has no authentication and should remain local or behind an authenticated gateway.

## License

MIT. See LICENSE.
