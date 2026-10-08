import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CSV_NAME = "calculated_data.csv"

if __name__ == "__main__":
    ##-- Visualisation --##
    #### ---- IC_pincount ---- ####
    # START Gemini (AI) helped this bit, this generates our third scatter parameter, frequency of appearing and makes a new dictionary (grouped) that has the new information

    calculated_parameters = pd.read_csv(CSV_NAME)

    plt.figure("DFT lines per component")
    plt.title("DFT lines per component")
    counts = (
        calculated_parameters.groupby(["component_name", "fdt"])
        .size()
        .reset_index(name="frequency")
    )
    IC_pincount_plot = sns.scatterplot(x="component_name", y="fdt", 
                                       data=counts,
                                       hue="frequency",       
                                       size="frequency",      
                                       sizes=(40, 300),   
                                       palette="flare",)     

    plt.show()


    # # Grab parameters from file
    # component_name_list = calculated_parameters["component_name"].to_list()
    # IC_pincount_list = calculated_parameters["IC_pincount"].to_list()
    # unique_colors_list = calculated_parameters["unique_colors"].to_list()

    # # Seaborn theme / style
    # sns.set_theme(style="ticks", palette="pastel")


    # plt.figure(1)
    # plt.title("IC_Pinout per component")

    # counts = (
    #     calculated_parameters.groupby(["component_name", "IC_pincount"])
    #     .size()
    #     .reset_index(name="frequency")
    # )
    # IC_pincount_plot = sns.scatterplot(x="component_name", y="IC_pincount", 
    #                                    data=counts,
    #                                    hue="frequency",       
    #                                    size="frequency",      
    #                                    sizes=(40, 300),   
    #                                    palette="flare",)     



    # #### ---- END IC_pincount ---- ####

    # plt.figure(2)
    # plt.title("IC_Pinout per component")
    # unique_colors_plot = sns.boxplot(x="component_name", y="unique_colors",
    #                                 data=calculated_parameters)
    # unique_colors_plot.set_xlabel("Component")
    # unique_colors_plot.set_ylabel("unique colors [n]")
    # unique_colors_plot.set_title("Unique colors per component")
    # unique_colors_plot.tick_params(axis='x', rotation=45)

    # plt.show()