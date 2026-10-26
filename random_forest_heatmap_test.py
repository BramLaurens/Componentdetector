import sklearn as skl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

DATA_PATH = "calculated_data.csv"
TESTSET_N_PER_CLASS = 50  # Number of samples per class for the test set
RANDOM_STATE = 42

# A function that returns a string of formatted time.
def current_time():
    return datetime.now().strftime("%H:%M:%S.%f")


def generate_test_train():
    data = pd.read_csv(DATA_PATH)
    data_grouped = data.groupby('component_name')       # This sorts in shape (1, n) where n is the amount of this component in the data.
    test_indexes = []

    for name, group_df in data_grouped:
        df_filtered = data[data['component_name'] == name]

        # Safety
        if len(df_filtered) < TESTSET_N_PER_CLASS:
            print(f"Warning: Not enough samples for class '{name}'. Available: {amount}, Required: {TESTSET_N_PER_CLASS}. Skipping this class.")
            n_samples = len(df_filtered)
        else:
            n_samples = TESTSET_N_PER_CLASS


        sampled_subset = group_df.sample(n=n_samples, random_state=RANDOM_STATE)
        test_indexes.extend(sampled_subset.index.tolist())

    test_set = data.loc[test_indexes]
    train_set = data.drop(test_indexes)

    return test_set, train_set


if __name__ == "__main__":
    data = pd.read_csv(DATA_PATH)
    x = data.drop(columns=['photo_number', 'component_name'])
    y = data['component_name']

    print("{} - Samples loaded from {}, generating test and train set.".format(current_time(), DATA_PATH))
    # Generate test and train set and prepare x_train, y_train, x_test, y_test
    test_set, train_set = generate_test_train()

    x_train = train_set.drop(columns=['photo_number', 'component_name'])
    y_train = train_set['component_name']

    x_test = test_set.drop(columns=['photo_number', 'component_name'])
    y_test = test_set['component_name']
    print("{} - Test and train set generated. Training Random Forest.".format(current_time()))

    # random_forest = skl.ensemble.RandomForestClassifier(random_state=RANDOM_STATE)

    # train_starttime = datetime.now()
    # random_forest.fit(x_train, y_train)
    # print("{} - Random Forest trained in {} seconds.".format(current_time(), (datetime.now() - train_starttime).total_seconds()))
    
    # print("{} - Scorign Random Forest.".format(current_time()))
    # y_predicted = random_forest.predict(x_test)
    # accuracy = skl.metrics.accuracy_score(y_test, y_predicted)

    # print("{} - Random Forest Accuracy: {} %".format(current_time(), (round(accuracy * 100, 3))))
    accuracy_list = []
    test_amount_n = 20
    test_amount_depth = 20
    for i in range(test_amount_n):
        for j in range(test_amount_depth):
            decision_engine = skl.ensemble.RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=i+1, max_depth=j+1)
            decision_engine.fit(x_train, y_train)

            y_predicted = decision_engine.predict(x_test)
            accuracy = skl.metrics.accuracy_score(y_test, y_predicted)
            accuracy_list.append((i+1, j+1, accuracy))

        print("tested {} / {}".format(int(((i+1) / test_amount_n) * test_amount_n * test_amount_depth), test_amount_n * test_amount_depth))


    print("Best accuracy: {} % with n_estimators={} and max_depth={}".format(round(max(accuracy_list, key=lambda x: x[2])[2] * 100, 3), max(accuracy_list, key=lambda x: x[2])[0], max(accuracy_list, key=lambda x: x[2])[1]))

    accuraty_df = pd.DataFrame(accuracy_list, columns=['n_estimators', 'max_depth', 'accuracy'])
    pivot_table = accuraty_df.pivot(index='max_depth', columns='n_estimators', values='accuracy')

    heatmap_plot = sns.heatmap(
        pivot_table, 
        annot=True,          # Set to True if you want the actual numbers in the grid cells (might look crowded for 20x20)
        cmap="coolwarm",       # Color palette (e.g., 'viridis', 'coolwarm', 'Blues')
        cbar_kws={'label': 'Accuracy'}
    )

    heatmap_plot.figure.set_size_inches(16, 9)
    heatmap_plot.figure.canvas.manager.set_window_title("Different Random Forest parameter accuracy heatmap")
    heatmap_plot.set_title("Random Forest Accuracy Heatmap", fontsize=16)

    plt.show()
    # heatmap_plot.figure.subplots_adjust(top=0.80)