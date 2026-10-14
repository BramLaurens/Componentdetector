import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CSV_NAME = "calculated_data.csv"

# matplotlib font styling
PLT_FONT_SIZE = 15

# Seaborn theme / style
sns.set_theme(style="ticks", palette="pastel")


if __name__ == "__main__":
    ##-- Visualisation --##
    #### ---- IC_pincount ---- ####
    # START Gemini (AI) helped this bit, this generates our third scatter parameter, frequency of appearing and makes a new dictionary (grouped) that has the new information

    calculated_parameters = pd.read_csv(CSV_NAME)





    plt.figure("DFT lines per component", figsize=(16, 9), layout="constrained")
    plt.title("DFT lines per component")
    counts = (
        calculated_parameters.groupby(["component_name", "fft_lines"])
        .size()
        .reset_index(name="frequency")
    )
    dft_lines_plot = sns.boxplot(x="component_name", y="fft_lines", 
                                       data=counts)  

    dft_lines_plot.set_xlabel("")
    dft_lines_plot.set_ylabel("Aantal lijnen [n]")
    
    dft_lines_plot.set_title("Aantal lijnen in fft per component", fontsize=PLT_FONT_SIZE*1.5)
    dft_lines_plot.xaxis.set_label_text("")
    dft_lines_plot.yaxis.set_label_text("Aantal lijnen [n]", fontsize=PLT_FONT_SIZE*1.2)
    
    dft_lines_plot.tick_params(axis='x', labelsize=PLT_FONT_SIZE, rotation=45, labelrotation_mode='xtick')
    dft_lines_plot.tick_params(axis='y', labelsize=PLT_FONT_SIZE)   


    plt.figure("DFT opp per component", figsize=(16, 9), layout="constrained")
    plt.title("DFT opp per component")
    counts = (
        calculated_parameters.groupby(["component_name", "fft_opp"])
        .size()
        .reset_index(name="frequency")
    )
    dft_opp_plot = sns.boxplot(x="component_name", y="fft_opp", 
                                        data=counts)  

    dft_opp_plot.set_xlabel("")
    dft_opp_plot.set_ylabel("oppervlak [px]")
    
    dft_opp_plot.set_title("oppervlak in fft per component", fontsize=PLT_FONT_SIZE*1.5)
    dft_opp_plot.xaxis.set_label_text("")
    dft_opp_plot.yaxis.set_label_text("oppervlak [px]", fontsize=PLT_FONT_SIZE*1.2)
    
    dft_opp_plot.tick_params(axis='x', labelsize=PLT_FONT_SIZE, rotation=45, labelrotation_mode='xtick')
    dft_opp_plot.tick_params(axis='y', labelsize=PLT_FONT_SIZE)   
    

    


    # # Grab parameters from file
    # component_name_list = calculated_parameters["component_name"].to_list()
    # IC_pincount_list = calculated_parameters["IC_pincount"].to_list()
    # unique_colors_list = calculated_parameters["unique_colors"].to_list()



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

    # plt.figure("Unique Colors", figsize=(16, 9), layout="constrained")
    # plt.title("Unique colors per component")
    # unique_colors_plot = sns.boxplot(x="component_name", y="unique_colors",
    #                                 data=calculated_parameters)
    # unique_colors_plot.set_xlabel("")
    # unique_colors_plot.set_ylabel("unique colors [n]")
    
    # unique_colors_plot.set_title("Unieke kleuren per pixel per component", fontsize=PLT_FONT_SIZE*1.5)
    # unique_colors_plot.xaxis.set_label_text("")
    # unique_colors_plot.yaxis.set_label_text("Aantal kleuren [n]", fontsize=PLT_FONT_SIZE*1.2)
    
    # unique_colors_plot.tick_params(axis='x', labelsize=PLT_FONT_SIZE, rotation=45, labelrotation_mode='xtick')
    # unique_colors_plot.tick_params(axis='y', labelsize=PLT_FONT_SIZE)

    # plt.tight_layout()
    # plt.subplots_adjust(left=0.1)
    plt.show()