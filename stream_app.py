import pickle
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.express import choropleth_mapbox
from streamlit_option_menu import option_menu
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import plotly.graph_objs as go # visualization
from plotly.subplots import make_subplots
import plotly.figure_factory as ff # visualization
import plotly.offline as py
import os
import pandas as pd
from pandas import DataFrame

pd.set_option("display.max_columns", 50)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#plt.style.use("seaborn-dark")
sns.set_context("talk", font_scale=0.6)
from scipy.stats.contingency import chi2_contingency
from scipy.stats import normaltest, shapiro
import plotly.express as px
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.rcParams.update(
    {"lines.linewidth": 1, "patch.facecolor": "#ebe3df", "axes.facecolor": "#ebe3df"}
)

from sklearn.experimental import enable_iterative_imputer, enable_halving_search_cv
from sklearn.impute import IterativeImputer
from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedShuffleSplit,
    StratifiedKFold,
    GridSearchCV,
    cross_val_score,
    HalvingGridSearchCV,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
    FunctionTransformer,
    OrdinalEncoder,
)
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.decomposition import PCA
from sklearn import ensemble
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.compose import make_column_selector, ColumnTransformer
from imblearn.over_sampling import (
    KMeansSMOTE,
    ADASYN,
    RandomOverSampler,
    SMOTE,
    SVMSMOTE,
)
#from tqdm import tqdm

import plotly.graph_objs as go # visualization
from plotly.subplots import make_subplots
import plotly.figure_factory as ff # visualization
import plotly.offline as py
# import pandas_profiling
# from streamlit_pandas_profiling import st_profile_report
# from pydantic_settings import BaseSettings

model_file = 'model_C=1.0.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

# image = Image.open('images/atlantic.png')
# st.image(image,use_column_width=False)

def main():

	st.sidebar.info("Este aplicativo es creado para predecir el retiro voluntario de clientes en Atlantic Quantum Innovations. Si quiere ingresar los valores del cliente manualmente, seleccione la opción 'Online'. Si tiene un archivo con los datos del cliente, seleccione la opción 'Batch'")
	#st.sidebar.image(image2)
	st.title("Home")
	st.write("A continuación, se presentan los datos:")
	
def inicio():
      col1, col2 = st.columns(2)
      with col1:
           image3 = Image.open('images/logo.jpg')
           st.image(image3)
      with col2:
           image2 = Image.open('images/leaving.png')
           st.image(image2)
      image_atlantic = Image.open('images/atlantic.png')
      st.sidebar.image(image_atlantic)  
      st.header('''Customer Churn''', divider='rainbow')
      #st.divider()
      st.write("**Objetivo**")
      st.markdown('''El objetivo principal de este proyecto es desarrollar modelos estadísticos de predicción de la pérdida de clientes de telecomunicaciones. 
      ''')
      st.markdown("***Retención de clientes***")
      
      st.markdown('***Toma de decisiones basada en datos***')
      
      st.markdown("***La eficiencia operativa***.")

      st.markdown("""El conjunto de datos de abandono de servicios de telecomunicaciones contiene información sobre los clientes de una empresa de telecomunicaciones y si abandonaron (cancelaron su servicio) o no. Incluye varias variables, como datos demográficos del cliente (edad, sexo, estado civil, etc.) y datos de uso del servicio (número de servicios contratados, número de llamadas, minutos, método de facturación, valor de la mensualidad, etc).""")
      st.markdown("""Con estos datos se puede desarrollar modelos que puedan identificar clientes en riesgo y tomar medidas para evitar la deserción, lo que podría conducir a una mayor retención de clientes e ingresos para la empresa.
    """)
      st.markdown("""
      **Acerca de las características**

    ---
    - **customerID**: ID de cliente
    - **género**: si el cliente es hombre o mujer
    - **Mayor**: Si el cliente es persona mayor o no (1, 0)
    - **Pareja**: Si el cliente tiene pareja o no (Sí, No)
    - **Dependientes**: Si el cliente tiene dependientes o no (Sí, No)
    - **tenure**: Número de meses que el cliente ha permanecido en la empresa
    - **Servicio Telefónico**: Si el cliente tiene servicio telefónico o no (Sí, No)
    - **Múltiples Líneas**: Si el cliente tiene múltiples líneas o no (Sí, No, Sin servicio telefónico)
    - **Servicio de Internet**: Proveedor de servicio de Internet del cliente (DSL, Fibra óptica, No)
    - **Seguridad en línea**: Si el cliente tiene seguridad en línea o no (Sí, No, Sin servicio de Internet)
    - **Copia de seguridad en línea**: Si el cliente tiene copia de seguridad en línea o no (Sí, No, Sin servicio de Internet)
    - **Protección del dispositivo**: Si el cliente tiene protección del dispositivo o no (Sí, No, Sin servicio de Internet)
    - **Soporte Técnico**: Si el cliente tiene soporte técnico o no (Sí, No, Sin servicio de Internet)
    - **StreamingTV**: Si el cliente tiene streaming de TV o no (Sí, No, No servicio de internet)
    - **Streaming**: Si el cliente tiene películas en streaming o no (Sí, No, No hay servicio de Internet)
    - **Contrato**: El plazo del contrato del cliente (Mes a mes, Un año, Dos años)
    - **Facturación Electrónica**: Si el cliente tiene facturación electrónica o no (Sí, No)
    - **Método de pago**: El método de pago del cliente (cheque electrónico, cheque enviado por correo, transferencia bancaria (automática), tarjeta de crédito (automática))
    - **Cargos Mensuales**: El monto cobrado al cliente mensualmente
    - **Cargos Totales**: El monto total cobrado al cliente
    - **Etiqueta de abandono**: si el cliente abandonó o no (Sí o No)
        
        """)


def analisis():
    df=pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    #['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']
    st.write(df.head())
    st.markdown('')
    fig = px.bar(df, x='Churn')
    st.plotly_chart(fig, use_container_width=True)
    fig = px.box(df, x='Churn', y='tenure')
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Ver explicación"):
        st.write("""
            La gráfica de arriba muestra que existe una diferencia estadisticamente representativa
            entre la antiguedad del cliente y su propensión al abandono del servicio. Las personas con 
            mayor antiguedad son menos probables a renunciar al servicio, mientras que los más nuevos
            son mas propensos a cancelarlo.
        """)
    fig = px.pie(df, names='PaymentMethod')
    st.plotly_chart(fig, use_container_width=True)
    
    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns
    fig, axes = plt.subplots(2, 2, figsize=(7, 5))
    axes = axes.flatten()
    for i, column in enumerate(tqdm(numeric_columns)):
        ax = axes[i]
        sns.boxplot(data=df, x="Churn", y=column, ax=ax)
        ax.set_title(f"{column} vs Churn Label", fontsize=10)

        for k in ax.containers:
            ax.bar_label(
                k, fontsize=10, label_type="center", backgroundcolor="w", fmt="%.2f"
            )
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("""
    **Interpretación de resultados**

    ---
    #### Meses de tenencia:
    - Para los clientes que no han abandonado ("Churn Label" = "No"), la permanencia promedio es de aproximadamente 37,57 meses.
    - Para los clientes que han abandonado ("Churn Label" = "Sí"), la permanencia promedio es significativamente menor, alrededor de 17,98 meses.
    - Esto sugiere que los clientes que han permanecido más tiempo en la empresa tienen menos probabilidades de abandonar, como lo indica la menor permanencia promedio de los clientes abandonados.
    
    #### Cargos mensuales:
    - Para los clientes que no han abandonado, los "cargos mensuales promedio" son de aproximadamente "$61,27".
    - Para los clientes que han abandonado, los cargos mensuales promedio son ligeramente más altos, alrededor de "74,44 dólares".
    - Esto indica que los clientes que han abandonado tienden a tener cargos mensuales ligeramente más altos en promedio.
    """)
   
    corr = df.corr(numeric_only=True)

    #mask = np.triu(np.ones_like(corr, dtype=bool))

    fig= plt.figure(figsize=(6, 3))
    sns.heatmap(corr, annot=True, fmt=".2f", linecolor="c") #mask=mask,
    plt.title("Pearson's Correlation Matrix")
    st.pyplot(fig)


    
    #colors=["lightgreen", "red"]
    fig = px.pie(df, names='Churn')
    st.plotly_chart(fig, use_container_width=True)
    
    fig = px.scatter(df,x='MonthlyCharges',y='TotalCharges')
    st.plotly_chart(fig, use_container_width=True)

    #############################################
    total_charge = df["TotalCharges"]
    missing = total_charge[~total_charge.str.replace(".", "").str.isdigit()]
    df["TotalCharges"] = df["TotalCharges"].apply(pd.to_numeric, errors="coerce")
    
    
    df.drop(
        columns=["customerID"],
        inplace=True,
        errors="ignore",
    )

    imputer = IterativeImputer()
    df["TotalCharges"] = imputer.fit_transform(df[["TotalCharges"]])



    unique_counts = df.select_dtypes("O").nunique()
    binary_columns = unique_counts[unique_counts == 2].index.drop("Churn").tolist()
    categorical_columns = unique_counts[unique_counts > 2].index.tolist()
    target_column = "Churn"
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    #X_train.head(2)

    transformer = ColumnTransformer(
        [
            ("scaler", StandardScaler(), ["MonthlyCharges", "TotalCharges", "tenure"]),
            ("binary_encoder", OrdinalEncoder(), binary_columns),
            ("ohe", OneHotEncoder(drop="first"), categorical_columns),
        ],
        remainder="passthrough",
    )

    transformer.fit(X_train)
    columns = transformer.get_feature_names_out()
    columns = list(map(lambda x: str(x).split("__")[-1], columns))

    X_train = pd.DataFrame(transformer.transform(X_train), columns=columns)
    X_test = pd.DataFrame(transformer.transform(X_test), columns=columns)

    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    y_train = label_encoder.transform(y_train)
    y_test = label_encoder.transform(y_test)

    classifiers = {
        "lr": LogisticRegression(),
        "svc": SVC(),
        "knn": KNeighborsClassifier(),
        "dtree": DecisionTreeClassifier(),
        "bagging": ensemble.BaggingClassifier(),
        "rfc": ensemble.RandomForestClassifier(),
        "extra_trees": ensemble.ExtraTreesClassifier(),
        "adaboost": ensemble.AdaBoostClassifier(),
        "gb": ensemble.GradientBoostingClassifier(),
        "hgb": ensemble.HistGradientBoostingClassifier(),
        "nb": GaussianNB(),
        "nn": MLPClassifier(),
    }

    def evaluate_model(y_true, y_pred):
        all_metrics = {
            "acc": metrics.accuracy_score(y_true, y_pred),
            "precision": metrics.precision_score(y_true, y_pred),
            "recall": metrics.recall_score(y_true, y_pred),
            "f1": metrics.f1_score(y_true, y_pred),
        }
        return all_metrics
    
    model_performances = []

    for label, model in tqdm(classifiers.items()):
        n_splits = 3
        kf = StratifiedKFold(shuffle=True, n_splits=n_splits, random_state=42)
        performaces = np.zeros((n_splits, 4))

        X_values = X_train.values
        i = 0
        for train_idx, test_idx in kf.split(X_values, y_train):
            train_set = X_values[train_idx], y_train[train_idx]
            test_set = X_values[test_idx], y_train[test_idx]
            model.fit(*train_set)
            y_pred = model.predict(test_set[0])
            perf = evaluate_model(test_set[1], y_pred)
            performaces[i, :] = list(perf.values())
            i += 1
        model_performances.append(
            pd.Series(np.mean(performaces, axis=0), index=list(perf.keys()), name=label)
        )    
    performaces_df = pd.concat(model_performances, axis=1)

    avg_f1 = performaces_df.loc["f1"].mean()
    fig, ax = plt.subplots()
    performaces_df.T.plot(
        kind="bar",
        title="Performance of models",
        colormap=plt.cm.viridis,
        width=0.8,
        figsize=(10, 4),
        ax=ax,
    )
    ylim = ax.get_ylim()
    ax.set(ylim=(0, ylim[-1] + 0.06))
    ax.hlines(avg_f1, *ax.get_xlim(), ls="--", label="avg_f1", lw=1.2)
    ax.legend(
        loc="best",
        shadow=True,
        frameon=True,
        facecolor="inherit",
        bbox_to_anchor=(0.15, 0.01, 1, 1),
        title="Metrics",
    )
    st.pyplot(fig)
    ############################################

def prediccion():
	add_selectbox = st.sidebar.selectbox(
	"Como te gustaría predecir?",
	("Online", "Batch"))
	st.title("Prediciendo el Abandono")
	if add_selectbox == 'Online':
		gender = st.selectbox('Gender:', ['male', 'female'])
		seniorcitizen= st.selectbox(' Customer is a senior citizen:', [0, 1])
		partner= st.selectbox(' Customer has a partner:', ['yes', 'no'])
		dependents = st.selectbox(' Customer has  dependents:', ['yes', 'no'])
		phoneservice = st.selectbox(' Customer has phoneservice:', ['yes', 'no'])
		multiplelines = st.selectbox(' Customer has multiplelines:', ['yes', 'no', 'no_phone_service'])
		internetservice= st.selectbox(' Customer has internetservice:', ['dsl', 'no', 'fiber_optic'])
		onlinesecurity= st.selectbox(' Customer has onlinesecurity:', ['yes', 'no', 'no_internet_service'])
		onlinebackup = st.selectbox(' Customer has onlinebackup:', ['yes', 'no', 'no_internet_service'])
		deviceprotection = st.selectbox(' Customer has deviceprotection:', ['yes', 'no', 'no_internet_service'])
		techsupport = st.selectbox(' Customer has techsupport:', ['yes', 'no', 'no_internet_service'])
		streamingtv = st.selectbox(' Customer has streamingtv:', ['yes', 'no', 'no_internet_service'])
		streamingmovies = st.selectbox(' Customer has streamingmovies:', ['yes', 'no', 'no_internet_service'])
		contract= st.selectbox(' Customer has a contract:', ['month-to-month', 'one_year', 'two_year'])
		paperlessbilling = st.selectbox(' Customer has a paperlessbilling:', ['yes', 'no'])
		paymentmethod= st.selectbox('Payment Option:', ['bank_transfer_(automatic)', 'credit_card_(automatic)', 'electronic_check' ,'mailed_check'])
		tenure = st.number_input('Number of months the customer has been with the current telco provider :', min_value=0, max_value=240, value=0)
		monthlycharges= st.number_input('Monthly charges :', min_value=0, max_value=240, value=0)
		totalcharges = tenure*monthlycharges
		output= ""
		output_prob = ""
		input_dict={
				"gender":gender ,
				"seniorcitizen": seniorcitizen,
				"partner": partner,
				"dependents": dependents,
				"phoneservice": phoneservice,
				"multiplelines": multiplelines,
				"internetservice": internetservice,
				"onlinesecurity": onlinesecurity,
				"onlinebackup": onlinebackup,
				"deviceprotection": deviceprotection,
				"techsupport": techsupport,
				"streamingtv": streamingtv,
				"streamingmovies": streamingmovies,
				"contract": contract,
				"paperlessbilling": paperlessbilling,
				"paymentmethod": paymentmethod,
				"tenure": tenure,
				"monthlycharges": monthlycharges,
				"totalcharges": totalcharges
			}

		if st.button("Predict"):
			X = dv.transform([input_dict])
			y_pred = model.predict_proba(X)[0, 1]
			churn = y_pred >= 0.5
			output_prob = float(y_pred)
			output = bool(churn)
		st.success('Churn: {0}, Risk Score: {1}'.format(output, output_prob))

	if add_selectbox == 'Batch':
		file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

		if file_upload is not None:

			data = pd.read_csv(file_upload)
			st.write(data)
			#records = data.to_dict(orient='records')
			X = dv.transform(data.to_dict(orient='records'))
			y_pred = model.predict_proba(X)[0, 1]
			churn = y_pred >= 0.5
			churn = bool(churn)
			st.write(churn)
			st.write(y_pred)
menu = {
    'title': 'Menu',
    'items': { 
        'Inicio' : {
            'action': inicio,
            'item_icon': 'house',
        },
        'Análisis' : {
            'action': analisis,
            'item_icon': 'bar-chart',
            'submenu': {
                'title': None,
                'items': {
                    'Bar chart' : {'action': '', 'item_icon': 'bar-chart-fill', 'submenu': None},
                    'Geo map' : {'action': '', 'item_icon': 'globe-americas', 'submenu': None},
                    'Correlations' : {'action': '', 'item_icon': 'bar-chart-line-fill', 'submenu': None},
                    'Ranking' : {'action': '', 'item_icon': 'table', 'submenu': None},
                },
                'menu_icon': None,
                'default_index': 0,
                'with_view_panel': 'main',
                'orientation': 'horizontal'
                
            } 
        },
        'Predicción' : {
            'action': prediccion,
            'item_icon': 'people'
        },
    },
    'menu_icon': 'clipboard2-check-fill', 
    'default_index': 0,
    'with_view_panel': 'sidebar',
    'orientation': 'vertical'
    }
def show_menu(menu):
    def _get_options(menu):
        options = list(menu['items'].keys())
        return options
    def _get_icons(menu):
        icons = [v['item_icon'] for _k, v in menu['items'].items()]
        return icons
    kwargs = {
        'menu_title': menu['title'],
        'options': _get_options(menu),
        'icons': _get_icons(menu),
        'menu_icon': menu['menu_icon'],
        'default_index': menu['default_index'],
        'orientation': menu['orientation']
    }
    with_view_panel = menu['with_view_panel']
    if with_view_panel == 'sidebar':
        with st.sidebar:
            menu_selection = option_menu(**kwargs)
    elif with_view_panel == 'main':
        menu_selection = option_menu(**kwargs)
    else:
        raise ValueError(f"Unknown view panel value: {with_view_panel}. Must be 'sidebar' or 'main'.")
    if 'submenu' in menu['items'][menu_selection] and menu['items'][menu_selection]['submenu']:
        show_menu(menu['items'][menu_selection]['submenu'])
    if 'action' in menu['items'][menu_selection] and menu['items'][menu_selection]['action']:
        menu['items'][menu_selection]['action']()
show_menu(menu)
