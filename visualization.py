import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CSV_NAME = "calculated_data.csv"

# Seaborn theme / style
sns.set_theme(style="ticks", palette="pastel")


if __name__ == "__main__":
    calculated_parameters = pd.read_csv(CSV_NAME)


    ### Pincount scatterplots and histograms ###
    pincount_components = ["DIP8", "BJT", "CerCapacitor", "Diode"]
    fig_scatter, ax_scatter = plt.subplots(2 ,2)
    fig_hist, ax_hist = plt.subplots(2 ,2)


    for i in range(len(pincount_components)):

        ## Scatterplots ## 
        counts = (
                calculated_parameters.groupby(["component_name", pincount_components[i] + "_pincount"])
                .size()
                .reset_index(name="frequency")
            )
        sns.scatterplot(
        data =          counts,
            x =         "component_name",
            y =         f"{pincount_components[i]}_pincount",
            hue =       "frequency",
            size =      "frequency",
            sizes =     (20, 300),
            palette =   "flare",
            legend =    False,
            ax =        ax_scatter[i // 2, i % 2]
            
        )
        ax_scatter[i // 2, i % 2].set_title("Pincount per component, optimized for {}".format(pincount_components[i]))
        ax_scatter[i // 2, i % 2].set_ylabel("Pin Count")
        ax_scatter[i // 2, i % 2].set_xlabel("")
        ax_scatter[i // 2, i % 2].tick_params(axis="x", rotation=30)


        ## Histograms ##
        histogram_data = calculated_parameters[ calculated_parameters["component_name"].str.contains(pincount_components[i], case=False, na=False) ]        

        sns.histplot(
            data =      histogram_data,
            x =         pincount_components[i] + "_pincount",
            discrete =  True,  # Keeps bins aligned cleanly with integer pin counts
            edgecolor = "black",
            ax =        ax_hist[i // 2, i % 2]
        )

        ax_hist[i // 2, i % 2].set_title("Pincount distribution ({})".format(pincount_components[i]))


        ## General plot settings
        plt.subplots_adjust(wspace=0, hspace=0)     


    fig_scatter.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.95, wspace=0.1, hspace=0.3)    
    fig_hist.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.95, wspace=0.1, hspace=0.3)    

    fig_scatter.canvas.manager.set_window_title("Pincount Scatterplots")


    ### END Pincount scatterplots and histograms ###


    ### Unique colors ###
    plt.figure("unique_colors")
    plt.title("Unique colors")
    unique_colors_plot = sns.boxplot(x="component_name", y="unique_colors",
                                    data=calculated_parameters)
    unique_colors_plot.set_xlabel("Component")
    unique_colors_plot.set_ylabel("unique colors [n]")
    unique_colors_plot.set_title("Unique colors per component")
    unique_colors_plot.tick_params(axis='x', rotation=45)

    ### END Unique colors ###

    plt.show()