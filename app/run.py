import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag, word_tokenize
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from sklearn.base import BaseEstimator, TransformerMixin

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
# Build a custom transformer which will extract the starting verb of a sentence
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class
    
    This class extract the starting verb of a sentence,
    creating a new feature for the ML classifier
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    # Given it is a tranformer we can return the self 
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)



# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse_table', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
#     category_names = df.iloc[:,5:].columns
#     category_boolean = (df.iloc[:,5:] != 0).sum().values
    
    rowsums = df.iloc[:,5:].sum(axis=1)
    multilabel_counts = rowsums.value_counts().sort_index()
    multi_labels, multi_label_counts = multilabel_counts.index, multilabel_counts.values
    
    pivot_data = df.groupby('genre').sum()[df.iloc[:,5:].columns]
    genre_names = list(pivot_data.index)
    

    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        # GRAPH 1 - genre graph
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
               # GRAPH 3 - Genre by category graph    
      {  'data': [{
                "type": "bar",
                    'x':multi_labels,
                    'y':multi_label_counts,
                    'marker_color':'indianred'
#                 
      } ],

            'layout': {
                'title': 'How many labels for each messages?',
                'yaxis': {
                    'title': "Number of Messages"
                },
                'xaxis': {
                    'title': "Number of labels"
                }
            }
        },
        # GRAPH 4 - Heat map   
      {  'data': [{
                "type": "heatmap",
                    'z': pivot_data.values.tolist(),
                     'x': pivot_data.columns.tolist(),
                     'y': pivot_data.index.tolist()
#                  #   marker_color='indianred'
#                 
      } ],

            'layout': {
                'title': 'Heat Map of Count of messages by Genre vs Category ',
                'yaxis': {
                    'title': "Genre",
                    'type': 'category'
                },
                'xaxis': {
                    'title': "Message Categories",
                    'type': 'category'
                }
            }
        }
    
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()