from flask import Flask, request
from predictor import main
import json
import traceback

app = Flask(__name__)

@app.route("/")
def hello():
	return json.dumps({"message":"Hello World", "statusCode":200})

@app.route("/predict/", methods=["GET", "POST"])
def accept():
	data = json.loads(request.get_data())
	try:
		results = [int(i) for i in main(data)]
		print(results)
		print(type(results))
		return json.dumps({"predictions":results, "statusCode":200})

	except Exception as e:
		print(traceback.print_exc())
		return json.dumps({"message":"Error Occured, please check your inputs", "statusCode":500})


if __name__ == "__main__":

	app.run(host="0.0.0.0", port=8002, debug=True)