import streamlit as st
st.title("Iris Classifier- API")

x1 = st.slider('Sepal Length', 4.3, 7.9, 0.5)
x2 = st.slider('Sepal Width', 2.0, 4.4, 0.5)
y1 = st.slider('Petal Length', 1.0, 6.9, 0.5)
y2 = st.slider('Petal Width', 0.1,2.5, 0.5)

from sklearn.datasets import load_iris
iris = load_iris()

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(iris.data,iris.target)

x = model.predict([[x1,x2,y1,y2]])
x = iris.target_names[x[0]]
st.title(x)
                               
