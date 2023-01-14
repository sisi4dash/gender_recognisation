from flask import Flask
from app import views

app = Flask(__name__) #webserver gatway

app.add_url_rule(rule='/',endpoint='home',view_func=views.home)
app.add_url_rule(rule='/home',endpoint='home',view_func=views.home)
app.add_url_rule(rule='/app/',endpoint='app',view_func=views.app)
app.add_url_rule(rule='/app/gender/',endpoint='gender',view_func=views.gender, methods=['GET','POST'])


if __name__ == "__main__":
    app.run(debug=True)