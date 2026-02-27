train:
	python src/train.py
tune: 
	python src/tune.py
register:
	python src/register.py --model_name fare-model --version 3 --stage Staging
	python src/register.py --model_name fare-model --version 3 --stage Production
serve:
	uvicorn src.serve_fastapi:app --host 0.0.0.0 --port 8000
test:
	python tests/test_api.py
batch:
	python src/batch_infer.py --input data/test.csv --output data/scored.csv --model-uri models:/fare-model/Production
