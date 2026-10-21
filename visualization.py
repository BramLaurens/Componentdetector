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
    calculated_parameters = pd.read_csv(CSV_NAME)

    # #### ---- DFT lines per component ---- ####
    # plt.figure("DFT lines per component", figsize=(16, 9), layout="constrained")
    # plt.title("DFT lines per component")
    # dft_lines_plot = sns.boxplot(x="component_name", y="fft_lines", 
    #                                    data=calculated_parameters)  
        
    # dft_lines_plot.set_title("Aantal lijnen in fft per component", fontsize=PLT_FONT_SIZE*1.5)
    # dft_lines_plot.xaxis.set_label_text("")
    # dft_lines_plot.yaxis.set_label_text("Aantal lijnen [n]", fontsize=PLT_FONT_SIZE*1.2)
    
    # dft_lines_plot.tick_params(axis='x', labelsize=PLT_FONT_SIZE, rotation=45, labelrotation_mode='xtick')
    # dft_lines_plot.tick_params(axis='y', labelsize=PLT_FONT_SIZE)   

    # #### ---- Unique Colors ---- ####
    # plt.figure("Unique Colors", figsize=(16, 9), layout="constrained")
    # plt.title("Unique colors per component")
    # unique_colors_plot = sns.boxplot(x="component_name", y="unique_colors",
    #                                 data=calculated_parameters)

    # unique_colors_plot.set_title("Unieke kleuren per pixel per component", fontsize=PLT_FONT_SIZE*1.5)
    # unique_colors_plot.xaxis.set_label_text("")
    # unique_colors_plot.yaxis.set_label_text("Aantal kleuren [n]", fontsize=PLT_FONT_SIZE*1.2)
    
    # unique_colors_plot.tick_params(axis='x', labelsize=PLT_FONT_SIZE, rotation=45, labelrotation_mode='xtick')
    # unique_colors_plot.tick_params(axis='y', labelsize=PLT_FONT_SIZE)


    # #### ---- Fill factor ---- ####
    # plt.figure("Fill factor Per component", figsize=(16, 9), layout="constrained")
    # plt.title("Fill factor Per component")
    # ff_plot = sns.boxplot(x="component_name", y="fill_factor", 
    #                                    data=calculated_parameters)  

    # ff_plot.set_title("Fill factor per component", fontsize=PLT_FONT_SIZE*1.5)
    # ff_plot.xaxis.set_label_text("")
    # ff_plot.yaxis.set_label_text("Fill factor [%]", fontsize=PLT_FONT_SIZE*1.2)

    # ff_plot.tick_params(axis='x', labelsize=PLT_FONT_SIZE, rotation=45, labelrotation_mode='xtick')
    # ff_plot.tick_params(axis='y', labelsize=PLT_FONT_SIZE)   

    ### ---- Body Aspect Rastio ---- ###
    plt.figure("Body Aspect Ratio Per Component", figsize=(16, 9 ), layout="constrained")
    plt.title("Body Aspect Ratio Per Component")
    ba_plot = sns.boxplot(x="component_name", y="body_aspect_ratio",
                                       data=calculated_parameters)


    #### ---- Fill factor ---- ####
    # plt.figure("Pairplot", figsize=(16, 9), layout="constrained")
    # plt.title("Pairplot")

    # data = calculated_parameters[["fft_lines", "fill_factor", "unique_colors", "component_name"]]
    # pairplot_plot = sns.pairplot(data, hue="component_name", palette="Set2", plot_kws={'s': 15})

    # pairplot_plot.figure.set_size_inches(16, 9)
    # pairplot_plot.figure.canvas.manager.set_window_title("Component Parameter Pairplot")
    # pairplot_plot.figure.subplots_adjust(top=0.80)

    # if pairplot_plot._legend:
    #     pairplot_plot._legend.remove()

    # handles = pairplot_plot._legend_data.values()
    # labels = pairplot_plot._legend_data.keys()


    # pairplot_legend = pairplot_plot.figure.legend(
    #     handles=handles, 
    #     labels=labels, 
    #     loc="upper center", 
    #     bbox_to_anchor=(0.5, 0.94), 
    #     ncol=7,  # 4 columns fit nicely horizontally across a 16x9 window
    #     frameon=False
    # )

    # for handle in pairplot_legend.legend_handles:
    #     if hasattr(handle, "set_marker"):
    #         handle.set_marker("s")
    #         handle.set_markersize(10)
    
    plt.tight_layout()
    # plt.subplots_adjust(left=0.1)
    plt.show()