import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def loadCsvFile(filepath):
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded {filepath}")
        return df
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found. Please check the path.")
        sys.exit(1)


@st.cache_data
def loadData():
    data = loadCsvFile('diabetes.csv')
    return data

@st.cache_resource
def prepareAndTrainModels(data):
    # Features and target
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models
    log_reg = LogisticRegression()
    rf = RandomForestClassifier()

    log_reg.fit(X_train_scaled, y_train)
    rf.fit(X_train_scaled, y_train)

    # Generate predictions
    log_reg_pred = log_reg.predict(X_test_scaled)
    rf_pred = rf.predict(X_test_scaled)

    return {
        'models': (log_reg, rf),
        'scaler': scaler,
        'data_split': (X_train, X_test, y_train, y_test),
        'predictions': (log_reg_pred, rf_pred)
    }

data = loadData()
model_data = prepareAndTrainModels(data)

# Sidebar for navigation
st.sidebar.title('Navigation')
page = st.sidebar.radio('Go to', ['EDA', 'Model Evaluation', 'Prediction'])

match page:
    case 'EDA':
        st.title("Exploratory Data Analysis - Diabetes Dataset")

        st.write("### Dataset Overview")
        st.write(f"Shape is -> {data.shape}")
        st.write("Columns and their Types are ->")
        
        column_info = pd.DataFrame({'Type': data.dtypes}).reset_index()
        column_info.columns = ['Column', 'Type']
        st.dataframe(column_info)

        # Distribution plots
        st.write("### Feature Distributions")
        feature_to_plot = st.selectbox(
            "Select feature to visualize:",
            data.drop('Outcome', axis=1).columns
        )

        fig = px.histogram(data, x=feature_to_plot, color='Outcome',
                        marginal="box",
                        nbins=30,
                        title=f"Distribution of {feature_to_plot} by Diabetes Outcome",
                        labels={'Outcome': 'Diabetes Status'})
        st.plotly_chart(fig)

        # Correlation heatmap
        st.write("### Feature Correlations")
        corr = data.corr()
        fig_corr = px.imshow(corr,
                            title="Correlation Heatmap",
                            labels=dict(color="Correlation"))
        st.plotly_chart(fig_corr)

        # Scatter plot
        st.write("### Feature Relationships")
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox('Select X-axis feature:', data.drop('Outcome', axis=1).columns)
        with col2:
            y_axis = st.selectbox('Select Y-axis feature:',
                                [col for col in data.drop('Outcome', axis=1).columns if col != x_axis])

        fig_scatter = px.scatter(data, x=x_axis, y=y_axis, color='Outcome',
                            title=f"{x_axis} vs {y_axis} by Diabetes Status",
                            labels={'Outcome': 'Diabetes Status'})
        st.plotly_chart(fig_scatter)
    case 'Model Evaluation':
        st.title("Model Evaluation")

        # Unpack data
        _, X_test, _, y_test = model_data['data_split']
        log_reg_pred, rf_pred = model_data['predictions']

        # Calculate metrics
        models = ['Logistic Regression', 'Random Forest']
        predictions = [log_reg_pred, rf_pred]

        # Create metrics comparison
        st.write("### Model Performance Metrics")

        metrics_data = []
        for model_name, y_pred in zip(models, predictions):
            metrics_data.append({
                'Model': model_name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1 Score': f1_score(y_test, y_pred)
            })

        metrics_df = pd.DataFrame(metrics_data)

        # Display metrics as a styled table
        st.dataframe(metrics_df.style.format({
            'Accuracy': '{:.3f}',
            'Precision': '{:.3f}',
            'Recall': '{:.3f}',
            'F1 Score': '{:.3f}'
        }))

        # Create confusion matrices
        st.write("### Confusion Matrices")

        # Create subplots for confusion matrices
        fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['Logistic Regression Confusion Matrix',
                                        'Random Forest Confusion Matrix'])

        for idx, (model_name, y_pred) in enumerate(zip(models, predictions), 1):
            cm = confusion_matrix(y_test, y_pred)

            # Create heatmap
            heatmap = go.Heatmap(
                z=cm,
                x=['Predicted Negative', 'Predicted Positive'],
                y=['Actual Negative', 'Actual Positive'],
                text=cm,
                texttemplate="%{z}",
                textfont={"size": 16},
                colorscale='Blues'
            )

            fig.add_trace(heatmap, row=1, col=idx)

        fig.update_layout(height=500)
        st.plotly_chart(fig)

        # ROC Curves
        st.write("### ROC Curves")

        from sklearn.metrics import roc_curve, auc

        # Calculate ROC curves
        fig_roc = go.Figure()

        for model_name, model in zip(models, model_data['models']):
            y_score = model.predict_proba(model_data['scaler'].transform(X_test))[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_score)
            auc_score = auc(fpr, tpr)

            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                name=f'{model_name} (AUC = {auc_score:.3f})',
                mode='lines'
            ))

        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            name='Random Chance',
            mode='lines',
            line=dict(dash='dash', color='gray')
        ))

        fig_roc.update_layout(
            title='ROC Curves',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
            width=700,
            height=500
        )

        st.plotly_chart(fig_roc)
    case 'Prediction':
        st.title("Diabetes Risk Prediction")

        # Unpack models and scaler
        log_reg, rf = model_data['models']
        scaler = model_data['scaler']

        # User input for all features
        st.write("### Enter Your Information")

        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.slider("Number of Pregnancies", 0, 17, 3)
            glucose = st.slider("Glucose Level (mg/dL)", 0, 200, 120)
            blood_pressure = st.slider("Blood Pressure (mm Hg)", 0, 122, 70)
            skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 20)

        with col2:
            insulin = st.slider("Insulin Level (mu U/ml)", 0, 846, 79)
            bmi = st.slider("BMI", 0.0, 67.1, 31.4)
            diabetes_pedigree = st.slider("Diabetes Pedigree Function", 0.078, 2.42, 0.3725)
            age = st.slider("Age (years)", 21, 81, 33)

        # Prepare input for prediction
        user_input = np.array([[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age
        ]])

        # Scale the input
        user_input_scaled = scaler.transform(user_input)

        # Predictions
        log_reg_pred = log_reg.predict_proba(user_input_scaled)[:, 1][0]
        rf_pred = rf.predict_proba(user_input_scaled)[:, 1][0]

        # Display results
        st.write("### Prediction Results")

        # Calculate predictions
        # Calculate average probability
        avgProb = (log_reg_pred + rf_pred) / 2

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Logistic Regression Probability",
                f"{log_reg_pred:.2%}"
            )
        with col2:
            st.metric(
                "Random Forest Probability",
                f"{rf_pred:.2%}"
            )

        # Risk Assessment and Recommendations
        st.write("### Risk Assessment and Recommendations")

        match avgProb:
            case _ if avgProb < 0.1:
                st.success("🟢 The model predicts you are in great shape keep up the good work! (< 10%)")
            case _ if avgProb > 0.1 and avgProb < 0.2:
                st.info("🔵 The model predicts there is a slight risk of diabetes. Maintaining a healthy diet and regular check-ups are advised! (<20%)")
            case _ if avgProb > 0.2 and avgProb < 0.3:
                st.info("🔵 The model predicts Moderate Low Risk (20-30%): Some lifestyle adjustments recommended")
            case _ if avgProb > 0.3 and avgProb < 0.4:
                st.info("🔵 The model predicts Moderate Risk (30-40%): Preventive measures should be taken")
            case _ if avgProb > 0.4 and avgProb < 0.5:
                st.warning("🟡 The model predicts Elevated Risk (40-50%): Medical attention recommended")
            case _ if avgProb > 0.5 and avgProb < 0.6:
                st.warning("🟡 The model predicts High Risk (50-60%): Immediate lifestyle changes needed")
            case _ if avgProb > 0.6 and avgProb < 0.7:
                st.warning("🟠 The model predicts Very High Risk (60-70%): Immediate medical attention required")
            case _ if avgProb > 0.7 and avgProb < 0.8:
                st.error("🔴 The model predicts Critical Risk (70-80%): Urgent medical intervention needed")
            case _:
                st.error("🔴 The model predicts Severe Risk (>80%): Emergency medical attention required")

        # Display the numerical risk percentage
        st.write(f"### Overall Risk Percentage based on the average logistical regression and random forest: {avgProb:.1%}")

        # Feature importance plot (for Random Forest)
        st.write("### Feature Importance")
        feature_importance = pd.DataFrame({
            'feature': data.drop('Outcome', axis=1).columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=True)

        fig_importance = px.bar(feature_importance, x='importance', y='feature',
                            orientation='h',
                            title='Feature Importance in Prediction')
        st.plotly_chart(fig_importance)
    case _:
        raise RuntimeError("Page not supported")
