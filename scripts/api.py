import logging

import flask
from flasgger import Swagger
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from predict import run_predict





logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)



# Initialize the Flask application
application = Flask(__name__)

application.config['ALLOWED_EXTENSIONS'] = set(['pdf'])
application.config['CONTENT_TYPES'] = {"pdf": "application/pdf"}
application.config["Access-Control-Allow-Origin"] = "*"


CORS(application)

swagger = Swagger(application)

def clienterror(error):
    resp = jsonify(error)
    resp.status_code = 400
    return resp


def notfound(error):
    resp = jsonify(error)
    resp.status_code = 404
    return resp


@application.route('/v1/character', methods=['POST'])
def character_classification():
    """run character classifiation prediction
        ---
        parameters:
          - name: body
            in: body
            schema:
              id: text
              required:
                - text
              properties:
                text:
                  type: string
            description: the required text for POST method
            required: true
        definitions:
          SentimentResponse:
          Project:
            properties:
              status:
                type: string
              ml-result:
                type: object
        responses:
          40x:
            description: Client error
          200:
            description: Character Classification Response
            examples:
                          [
{
  "status": "success",
  "sentiment": "1"
},
{
  "status": "error",
  "message": "Exception caught"
},
]
        """
    json_request = request.get_json()
    if not json_request:
        return Response("No json provided.", status=400)
    text = json_request['text']
    if text is None:
        return Response("No text provided.", status=400)
    else:
        label = run_predict(text)
        return flask.jsonify({"status": "success", "label": label})


@application.route('/v1/character/categories', methods=['GET'])
def character_categories():
    """Possible character categories.
        ---
        definitions:
          CategoriestResponse:
          Project:
            properties:
              categories:
                type: object
        responses:
          40x:
            description: Client error
          200:
            description: Character Classification Response
            examples:
                          [
{
  "categories": [1,2,3],
  "sentiment": "1"
}
]
        """
    return flask.jsonify({"categories": ['elaine','george','jerry','kramer']})


if __name__ == '__main__':
    # run the prediction function as a API REST service
    application.run(debug=True, use_reloader=True)
