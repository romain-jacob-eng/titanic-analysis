# Project 8 — Titanic Survival Analysis
# Analyzes the Titanic dataset, visualizes key patterns,
# and trains a classification model to predict survival.

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def load_data(filepath):
    df = pd.read_csv(filepath)
    return df


def explore_data(df):
    print(f"Dataset shape: {df.shape}")
    print(df.head())
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Survival counts:\n{df['Survived'].value_counts()}")


def visualize_data(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Survival rate by sex
    df.groupby('Sex')['Survived'].mean().plot(kind='bar', ax=axes[0])
    axes[0].set_title('Survival Rate by Sex')
    axes[0].set_xlabel('Sex')
    axes[0].set_ylabel('Survival Rate')

    # Survival rate by class
    df.groupby('Pclass')['Survived'].mean().plot(kind='bar', ax=axes[1])
    axes[1].set_title('Survival Rate by Class')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Survival Rate')

    # Age distribution
    df['Age'].hist(bins=30, ax=axes[2])
    axes[2].set_title('Age Distribution')
    axes[2].set_xlabel('Age')
    axes[2].set_ylabel('Count')

    plt.tight_layout()
    plt.show()


def prepare_features(df):
    df = df.copy()
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    return X, y


def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    classification = classification_report(y_test, predictions)
    print(f"Model accuracy: {accuracy * 100:.1f}%")
    print("Classification Report:")
    print(classification)


if __name__ == "__main__":
    filepath = os.path.join(os.path.dirname(__file__), "titanic.csv")
    df = load_data(filepath)
    explore_data(df)
    visualize_data(df)
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)