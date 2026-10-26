import sklearn as skl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import ConfusionMatrixDisplay

DATA_PATH = "calculated_data.csv"
TESTSET_N_PER_CLASS = 50  # Number of samples per class for the test set
RANDOM_STATE = 42

DISPLAY_CONFUSION_MATRIX = True

# A function that returns a string of formatted time.
def current_time():
    return datetime.now().strftime("%H:%M:%S.%f")


if __name__ == "__main__":
    # Load the training data from CSV and split into features and labels
    data = pd.read_csv(DATA_PATH)
    x = data.drop(columns=['photo_number', 'component_name'])
    y = data['component_name']

    # Use sklearn's train_test_split to create a test set with TESTSET_N_PER_CLASS samples per class
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE)
    print("{} - Test and train set generated (train_test_split). Training random forest.".format(current_time()))

    # Initialize RF classifier with default params
    random_forest = skl.ensemble.RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100, max_depth=None)        # Default n_estimators and max_depth values

    train_starttime = datetime.now()
    # Train the random forest classifier on the training data
    random_forest.fit(x_train, y_train)
    # Just some logging to show how long the training took
    print("{} - Random forest trained in {} seconds.".format(current_time(), (datetime.now() - train_starttime).total_seconds()))
    
    print("{} - Scoring random forest.".format(current_time()))

    # Score the random forest on the test set and print the accuracy and f1 score
    y_predicted = random_forest.predict(x_test)
    # Calculate accuracy and f1 score, compare predicted and actual labels (20/80)
    accuracy = skl.metrics.accuracy_score(y_test, y_predicted)
    # Calculate the f1 score using the weighted average method
    f1 = skl.metrics.f1_score(y_predicted, y_test, average='weighted')
    print("{} - Random rorest accuracy: {} %, with an f1 score of: {}.".format(current_time(), (round(accuracy * 100, 3)), round(f1, 3)))

    # Display the confusion matrix if DISPLAY_CONFUSION_MATRIX is set to True
    if DISPLAY_CONFUSION_MATRIX:
        # Display the confusion matrix using sklearn's ConfusionMatrixDisplay
        ConfusionMatrixDisplay.from_predictions(y_test, y_predicted, display_labels=random_forest.classes_, cmap=plt.cm.Blues)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

    