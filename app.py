from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import io
import base64

#falsk app
app = Flask(__name__)

data = pd.read_csv("train - Walmart Sales Forecast.csv")
data['Date'] = pd.to_datetime(data['Date'])
data['Week'] = data['Date'].dt.isocalendar().week.astype(int)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    graph = None
    error = None
    if request.method == "POST":
        product = int(request.form["product"])
        week = int(request.form["week"])
        product_data = data[data['Dept'] == product]
        if product_data.empty:
            error = "Product not found; please try another."
        else:
            X = product_data[['Week']]
            y = product_data['Weekly_Sales']
            
            model = LinearRegression()
            model.fit(X, y)
            prediction = model.predict([[week]])[0]
            
            plt.figure()
            plt.scatter(X, y)
            plt.plot(X, model.predict(X))
            plt.xlabel("Week")
            plt.ylabel("Weekly Sales")
            plt.title(f"Sales Trend for Product {product}")
            
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            graph = base64.b64encode(img.getvalue()).decode()
            plt.close()
    return render_template(
        "index.html",
        prediction=prediction,
        graph=graph,
        error=error
    )
if __name__ == "__main__":
                app.run(host="0.0.0.0", port=7860)